"""End-to-end test for induction/pipeline.py (module 3.7: orchestration)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.core.modeling_schemas import (
    ModelingExperience,
)
from or_experience_bank.core.modeling_store import ModelingStore
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.induction.alignment import LLMBackedAligner, StructuralAligner
from or_experience_bank.induction.candidates import SignatureClusterer
from or_experience_bank.induction.counterexample import (
    CounterexampleSearcher,
    LLMBackedCounterexampleSearcher,
)
from or_experience_bank.induction.inducer import LLMBackedInducer, PatternInducer
from or_experience_bank.induction.pipeline import InductionPipeline, run_induction_sync
from or_experience_bank.induction.validation import PatternValidator
from or_experience_bank.llm_client import FakeLLMClient


_FAMILY_INDEX = {
    "e1": "inventory",
    "e2": "scheduling",
    "e3": "workforce",
}


def make_realization(exp_id, method_text):
    rec = ModelingExperience(
        title=method_text,
        retrieval_text=method_text,
    )
    rec.math_scope.structural_signature = rec.math_scope.structural_signature.from_dict({
        "objective": "linear",
        "decision": ["binary_assignment"],
        "constraint": ["capacity"],
        "interaction": "shared_resource_coupled",
        "features": {"resource": "shared_scarce"},
    })
    rec.method.action_template = method_text
    rec.evidence.source_episodes = ["ep_" + exp_id]
    rec.experience_id = exp_id
    rec.compute_content_hash()
    return rec


def resolver(record):
    # family resolved from the test's episode index via provenance, not a schema field
    for ep in (record.get("evidence") or {}).get("source_episodes", []):
        rid = ep.replace("ep_", "")
        if rid in _FAMILY_INDEX:
            return _FAMILY_INDEX[rid]
    return record.get("experience_id", "")


class StubExecutor:
    async def execute(self, code_path, workspace, solver):
        # refutation program "runs" and reports the principle does NOT fail
        return SolverExecutionResult(
            status="ok", solver=solver, stdout='{"principle_failed": false}'
        )


def build_store(tmp):
    store = ModelingStore(Path(tmp))
    for exp_id, text in [
        ("e1", "allocate stock x[i] to warehouse capacity"),
        ("e2", "assign jobs x[j] to machine capacity"),
        ("e3", "assign workers x[k] to labor-hour capacity"),
    ]:
        store.append(make_realization(exp_id, text))
    return store


def build_llm():
    alignment = {
        "roles": ["resource_pool", "competing_decisions", "objective_contribution"],
        "bindings": [
            {"realization_id": "e1", "problem_family": "inventory",
             "mapping": {"resource_pool": "warehouse", "competing_decisions": "stock"}},
            {"realization_id": "e2", "problem_family": "scheduling",
             "mapping": {"resource_pool": "machine", "competing_decisions": "jobs"}},
            {"realization_id": "e3", "problem_family": "workforce",
             "mapping": {"resource_pool": "labor", "competing_decisions": "workers"}},
        ],
        "confidence": 0.9,
    }
    hypothesis = [{
        "statement": "When decisions compete for a shared scarce resource with marginal "
                     "contribution, prioritize higher marginal contribution subject to coupling.",
        "structural_pattern": "shared scarce resource allocation",
        "roles_used": ["resource_pool", "competing_decisions"],
        "applicability_conditions": ["linear objective"],
        "complexity": 0.3,
    }]
    failure_conditions = [["fixed setup cost"]]
    refutation_program = ["print('check')"]
    # object order: (1) alignment, (2) induced hypotheses, (3) failure conditions
    return FakeLLMClient(
        text_responses=refutation_program,
        object_responses=[alignment, hypothesis] + failure_conditions,
    )


async def transfer_solver(task, principle):
    # principle helps on the unseen task
    return 8.0 if principle is not None else 12.0


class InductionPipelineTest(unittest.TestCase):
    def test_end_to_end_validates_and_appends_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = build_store(tmp)
            self.assertEqual(len(store.all_records()), 3)
            llm = build_llm()
            pipeline = InductionPipeline(
                store=store,
                clusterer=SignatureClusterer(family_resolver=resolver),
                aligner=LLMBackedAligner(StructuralAligner(), llm),
                inducer=LLMBackedInducer(PatternInducer(), llm),
                counterexample=LLMBackedCounterexampleSearcher(
                    CounterexampleSearcher(executor=StubExecutor()), llm
                ),
                validator=PatternValidator(validation_threshold=0.5),
                transfer_solver=transfer_solver,
                workspace=Path(tmp) / "ws",
                unseen_tasks=["unseen allocation task"],
            )
            report = run_induction_sync(pipeline)
            self.assertEqual(report.clusters_found, 1)
            self.assertEqual(report.hypotheses_generated, 1)
            self.assertEqual(report.patterns_validated, 1)
            self.assertEqual(report.patterns_refuted, 0)

            # pattern appended as peer, sources untouched
            patterns = store.validated_records()
            self.assertEqual(len(patterns), 1)
            p = patterns[0]
            self.assertEqual(p["status"], "validated")
            self.assertTrue(p.get("method", {}).get("action_template", ""))
            self.assertEqual(sorted(p["derived_from_experience_ids"]), ["e1", "e2", "e3"])
            self.assertIn("resource_pool", p["role_schema"])
            self.assertEqual(len(p["role_mappings"]), 3)
            # sources preserved (append-only) + 1 new induced record = 4 total
            self.assertEqual(len(store.all_records()), 4)

    def test_refuted_when_no_transfer_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = build_store(tmp)
            llm = build_llm()

            async def no_help(task, principle):
                return 10.0  # principle gives no improvement

            pipeline = InductionPipeline(
                store=store,
                clusterer=SignatureClusterer(family_resolver=resolver),
                aligner=LLMBackedAligner(StructuralAligner(), llm),
                inducer=LLMBackedInducer(PatternInducer(), llm),
                counterexample=LLMBackedCounterexampleSearcher(
                    CounterexampleSearcher(executor=StubExecutor()), llm
                ),
                validator=PatternValidator(validation_threshold=0.5),
                transfer_solver=no_help,
                workspace=Path(tmp) / "ws",
                unseen_tasks=["unseen task"],
            )
            report = run_induction_sync(pipeline)
            # Induction != Summary guard: no unseen transfer -> refuted, NOT appended.
            self.assertEqual(report.patterns_validated, 0)
            self.assertEqual(report.patterns_refuted, 1)
            self.assertEqual(store.validated_records(), [])  # refuted NOT appended to bank
            self.assertEqual(len(report.refuted), 1)  # but archived in report

    def test_no_clusters_no_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelingStore(Path(tmp))  # empty
            pipeline = InductionPipeline(
                store=store,
                clusterer=SignatureClusterer(family_resolver=resolver),
                aligner=LLMBackedAligner(llm_client=None),
                inducer=LLMBackedInducer(llm_client=None),
                counterexample=LLMBackedCounterexampleSearcher(llm_client=None),
                validator=PatternValidator(),
            )
            report = run_induction_sync(pipeline)
            self.assertEqual(report.clusters_found, 0)
            self.assertEqual(report.patterns_validated, 0)


if __name__ == "__main__":
    unittest.main()
