"""Round-trip and validation tests for Phase 0 modeling schemas (steps 1-3)."""

import unittest

from or_experience_bank.core.modeling_schemas import (
    CONSTRAINT_STRUCTURES,
    DECISION_STRUCTURES,
    INTERACTION_COUPLINGS,
    MODELING_ASPECTS,
    OBJECTIVE_STRUCTURES,
    BranchSummary,
    EpisodeRecord,
    ModelingExperience,
    SignatureValidationError,
    StructuralSignature,
)


def _cvrp_signature() -> StructuralSignature:
    return StructuralSignature(
        objective="linear",
        decision=["binary_assignment", "multi_index_3d"],
        constraint=["capacity", "flow_conservation"],
        interaction="shared_resource_coupled",
        features={"network": "path_on_graph", "resource": "shared_scarce"},
    )


class StructuralSignatureTest(unittest.TestCase):
    def test_valid_signature_round_trip(self):
        signature = _cvrp_signature().validate()
        restored = StructuralSignature.from_dict(signature.to_dict())
        self.assertEqual(restored, signature)

    def test_invalid_core_value_rejected(self):
        with self.assertRaises(SignatureValidationError):
            StructuralSignature(objective="not_a_real_objective").validate()
        with self.assertRaises(SignatureValidationError):
            StructuralSignature(decision=["mystery_var"]).validate()
        with self.assertRaises(SignatureValidationError):
            StructuralSignature(constraint=["not_a_constraint"]).validate()
        with self.assertRaises(SignatureValidationError):
            StructuralSignature(interaction="telepathy").validate()

    def test_open_features_not_validated(self):
        # Open feature slots accept arbitrary keys/values without raising.
        signature = StructuralSignature(features={"uncertainty": "scenario_tree", "brand_new_dim": "x"})
        self.assertEqual(signature.validate().features["brand_new_dim"], "x")

    def test_math_type_summary_derived_not_stored(self):
        summary = _cvrp_signature().math_type_summary()
        self.assertIn("shared_resource_coupled", summary)
        self.assertIn("linear", summary)

    def test_core_key_order_insensitive_for_multivalue(self):
        a = StructuralSignature(constraint=["capacity", "covering"], decision=["binary_assignment"])
        b = StructuralSignature(constraint=["covering", "capacity"], decision=["binary_assignment"])
        self.assertEqual(a.core_key(), b.core_key())

    def test_shared_feature_keys_intersection_only(self):
        a = StructuralSignature(features={"network": "path_on_graph", "temporal": "single_period"})
        b = StructuralSignature(features={"network": "path_on_graph", "resource": "shared_scarce"})
        self.assertEqual(a.shared_feature_keys(b), ["network"])

    def test_vocabularies_nonempty(self):
        for vocab in (OBJECTIVE_STRUCTURES, DECISION_STRUCTURES, CONSTRAINT_STRUCTURES, INTERACTION_COUPLINGS):
            self.assertTrue(len(vocab) >= 4)


class ModelingExperienceTest(unittest.TestCase):
    def _realization(self) -> ModelingExperience:
        record = ModelingExperience(
            title="Use inventory balance to couple consecutive periods",
            retrieval_text="multi-period inventory balance constraint",
            modeling_aspect="constraint",
        )
        record.math_scope.structural_signature = StructuralSignature(
            objective="linear",
            decision=["continuous_flow"],
            constraint=["flow_conservation"],
            interaction="shared_resource_coupled",
            features={"temporal": "multi_period_balance"},
        )
        record.method.action_template = "Introduce I_t and impose I_t = I_(t-1) + in_t - out_t"
        record.method.wrong_form = "Replacing balance equality with <="
        record.evidence.source_episodes = ["ep_001"]
        return record

    def test_realization_round_trip(self):
        record = self._realization().validate()
        restored = ModelingExperience.from_dict(record.to_dict())
        self.assertEqual(restored.to_dict(), record.to_dict())
        self.assertIsNone(restored.status)

    def test_validated_record_round_trip(self):
        record = self._realization()
        record.status = "validated"
        record.role_schema = {"resource_pool": "shared scarce resource"}
        record.role_mappings = []
        record.scoring.total = 0.71
        self.assertEqual(record.validate().status, "validated")
        restored = ModelingExperience.from_dict(record.to_dict())
        self.assertEqual(restored.scoring.total, 0.71)

    def test_invalid_status_rejected(self):
        record = self._realization()
        record.status = "maybe"
        with self.assertRaises(ValueError):
            record.validate()

    def test_invalid_modeling_aspect_rejected(self):
        record = self._realization()
        record.modeling_aspect = "not_a_real_aspect"
        with self.assertRaises(ValueError):
            record.validate()

    def test_content_hash_stable_and_excludes_identity(self):
        record = self._realization()
        hash_a = record.compute_content_hash()
        record.experience_id = "exp_different"
        record.created_at = "1999-01-01T00:00:00+00:00"
        hash_b = record.compute_content_hash()
        self.assertEqual(hash_a, hash_b)  # identity/time do not affect semantic hash


class EpisodeRecordTest(unittest.TestCase):
    def test_episode_round_trip(self):
        episode = EpisodeRecord(
            problem=" replenish a warehouse weekly ",
            problem_id="prob_abc",
            final_objective=1234.5,
            gold_answer=1234.5,
            produced_realization_ids=["exp_1", "exp_2"],
        )
        episode.structural_signature = _cvrp_signature()
        episode.branches = [
            BranchSummary(solver="gurobi", status="optimal", attempts=2, objective_value=1234.5),
            BranchSummary(solver="ortools", status="timeout", attempts=3),
        ]
        restored = EpisodeRecord.from_dict(episode.to_dict())
        self.assertEqual(restored.to_dict(), episode.to_dict())
        self.assertEqual(len(restored.branches), 2)
        self.assertEqual(restored.branches[0].solver, "gurobi")


if __name__ == "__main__":
    unittest.main()
