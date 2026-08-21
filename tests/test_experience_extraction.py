from __future__ import annotations

import unittest
import asyncio

from helpers import SRC_DIR
from or_experience_bank.experience.extractor import ExperienceExtractor
from or_experience_bank.core.schemas import (
    AttemptRecord, BranchResult, SolverExecutionResult, ValidationReport,
)
from or_experience_bank.solving.validator import ExperienceValidator
from or_experience_bank.llm_client import FakeLLMClient


def attempt(number, error=None, status="error", level="runtime_only"):
    return AttemptRecord(
        attempt_id="att_{}".format(number), problem_id="prob_test", branch_id="gurobi-1", solver="gurobi",
        attempt_number=number, started_at="2026-01-01T00:00:00+00:00", finished_at="2026-01-01T00:00:01+00:00",
        retrieved_experience_ids={"modeling": [], "implementation": [], "repair": [], "solving": []},
        problem_summary="assignment", formulation_summary="binary assignment", code_path="runs/code.py", code_hash=str(number),
        normalized_error=error, solver_status=status, validation_level=level,
        repair_action_summary="Use quicksum instead of a Python list" if number == 2 else "",
    )


class ExtractionTests(unittest.TestCase):
    def test_error_before_success_extracts_solver_specific_repair(self):
        branch = BranchResult(
            branch_id="gurobi-1", solver="gurobi", workspace="runs/gurobi-1",
            attempts=[attempt(1, "TypeError: list is not a linear expression"), attempt(2, status="optimal", level="solver_feasible")],
            execution=SolverExecutionResult(status="optimal", solver="gurobi", exit_code=0, objective_sense="minimize", objective_value=1, variables={}),
            validation=ValidationReport(True, "solver_feasible"), termination_reason="optimal",
        )
        records = ExperienceExtractor().extract_intra_branch(branch, "assignment")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].layer, "repair")
        self.assertEqual(records[0].scope.generality, "solver_specific")
        self.assertEqual(records[0].scope.solver, "gurobi")

    def test_repeated_error_extracts_negative_and_vague_or_evidenceless_is_rejected(self):
        branch = BranchResult(
            branch_id="gurobi-1", solver="gurobi", workspace="runs/gurobi-1",
            attempts=[attempt(1, "same"), attempt(2, "same")],
            execution=SolverExecutionResult(status="error", solver="gurobi"),
            validation=ValidationReport(False), termination_reason="repeated_error",
        )
        records = ExperienceExtractor().extract_intra_branch(branch, "assignment")
        self.assertEqual(records[0].polarity, "negative")
        data = records[0].to_dict()
        from or_experience_bank.core.store import compute_content_hash
        data["policy"]["action"] = "check constraints"
        data["evidence"]["branch_ids"] = []
        data["evidence"]["attempt_ids"] = []
        data["content_hash"] = compute_content_hash(data)
        report = ExperienceValidator().validate(data)
        self.assertFalse(report.valid)
        self.assertTrue(any("vague" in error for error in report.errors))
        self.assertTrue(any("branch_ids" in error for error in report.errors))

    def test_two_solver_consistency_extracts_solver_agnostic_modeling(self):
        branches = []
        for solver in ("gurobi", "scip"):
            item = attempt(1, status="optimal", level="solver_feasible")
            item.branch_id = solver + "-1"
            item.solver = solver
            branches.append(BranchResult(
                branch_id=item.branch_id, solver=solver, workspace="runs/" + solver,
                attempts=[item],
                execution=SolverExecutionResult(status="optimal", solver=solver, exit_code=0, objective_sense="minimize", objective_value=5, variables={}),
                validation=ValidationReport(True, "solver_feasible"), termination_reason="optimal",
            ))
        records = ExperienceExtractor().extract_cross_branch(branches, "assignment")
        self.assertEqual(records[0].layer, "modeling")
        self.assertEqual(records[0].scope.generality, "solver_agnostic")

    def test_malformed_llm_extraction_is_retried_once_and_not_returned(self):
        client = FakeLLMClient(object_responses=["{}", "not json"])
        extractor = ExperienceExtractor(client)
        records = asyncio.run(extractor.extract_with_llm("extract strict records"))
        self.assertEqual(records, [])
        self.assertEqual(len(client.prompts), 2)
