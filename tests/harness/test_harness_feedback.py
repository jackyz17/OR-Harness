"""Tests for the harness-feedback fixes: model representation, coupling
derivation chain, cross-check warnings, pending staging, cross-execution C4,
and solver advisories. The SIRL_026 scenario (independent resource
constraints mis-supplied as rc=0.8) is reproduced end to end."""
import json
import textwrap
import unittest

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import CostVector, FailureRecord, StrategicEntry, Strategy
from or_harness.profiling.model_syntax import (
    coupling_from_model,
    parse_model,
    verify_model,
)
from or_harness.profiling.profiler import profile_task
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.triggers import (
    check_triggers,
    classify_failure,
    solver_advisories,
)
from or_harness.strategy.catalog import load_catalog


SIRL_MODEL = textwrap.dedent("""
    SETS:
     i in Projects = {p0, p1}
     j in Resources = {r0, r1}
    PARAMETERS:
     E[i,j]
     C[i,j]
     alpha
     limit[j]
    VARIABLES:
     x[i,j] continuous >= 0
    OBJECTIVE:
     maximize sum(i, sum(j, (E[i,j] - alpha * C[i,j]) * x[i,j]))
    CONSTRAINTS:
     C1: sum(i, x[i,j]) <= limit[j]
""")


class TestModelSyntax(HarnessTestCase):
    def test_parse_and_verify_valid_model(self):
        report = verify_model(SIRL_MODEL)
        self.assertTrue(report.passed, msg=[i.detail for i in report.issues])
        model = report.parsed
        self.assertEqual(len(model.variables), 1)
        self.assertEqual(len(model.constraints), 1)

    def test_l1_missing_block(self):
        broken = SIRL_MODEL.replace("CONSTRAINTS:", "NOT_A_BLOCK:")
        report = verify_model(broken)
        self.assertFalse(report.passed)
        self.assertTrue(any(i.code == "missing_block" for i in report.issues))

    def test_l2_undefined_symbol(self):
        # Reference an undeclared variable in a constraint (x stays declared;
        # z is never declared anywhere).
        broken = SIRL_MODEL.replace("C1: sum(i, x[i,j]) <= limit[j]",
                                    "C1: sum(i, z[i,j]) <= limit[j]")
        report = verify_model(broken)
        self.assertFalse(report.passed)
        self.assertTrue(any(i.code == "undefined_symbol" for i in report.issues))

    def test_l2_bad_label(self):
        broken = SIRL_MODEL.replace("C1:", "capacity:")
        report = verify_model(broken)
        self.assertFalse(report.passed)
        self.assertTrue(any(i.code == "bad_label" for i in report.issues))

    def test_coupling_independent_constraints_low(self):
        """The SIRL_026 ground truth: independent resource constraints ->
        resource_coupling 0, not 0.8."""
        model = parse_model(SIRL_MODEL)
        coupling = coupling_from_model(model)
        self.assertEqual(coupling["resource_coupling"], 0.0)

    def test_coupling_shared_constraint_high(self):
        coupled = SIRL_MODEL + " C2: sum(i, sum(j, x[i,j])) <= budget\n"
        model = parse_model(coupled)
        coupling = coupling_from_model(model)
        self.assertEqual(coupling["resource_coupling"], 1.0)  # x in both C1 and C2

    def test_temporal_from_set_names(self):
        temporal = textwrap.dedent("""
            SETS:
             t in Periods = {1, 2, 3}
             i in Items = {a, b}
            PARAMETERS:
             d[i,t]
            VARIABLES:
             x[i,t] continuous >= 0
            OBJECTIVE:
             minimize sum(t, sum(i, d[i,t] * x[i,t]))
            CONSTRAINTS:
             C1: sum(i, x[i,t]) >= 1
        """)
        coupling = coupling_from_model(parse_model(temporal))
        self.assertEqual(coupling["temporal_coupling"], 1.0)

    def test_semantic_never_derived(self):
        coupling = coupling_from_model(parse_model(SIRL_MODEL))
        self.assertIsNone(coupling["semantic_coupling"])


class TestProfilerChain(HarnessTestCase):
    def test_model_field_beats_supplied(self):
        task = {
            "task_id": "SIRL_026", "family": "allocation",
            "model": SIRL_MODEL,
            "annotations": {"coupling": {"resource_coupling": 0.8}},  # wrong
        }
        profile = profile_task(task)
        # Model derivation wins for structural dimensions.
        self.assertEqual(profile.resource_coupling, 0.0)
        # Cross-check warning fires (0.8 vs 0.0 cross bin boundaries).
        warnings = profile.annotations["profiling"]["coupling_warnings"]
        self.assertTrue(any(w["dimension"] == "resource_coupling" for w in warnings))

    def test_semantic_supplied_not_overridden(self):
        task = {"task_id": "t", "family": "allocation", "model": SIRL_MODEL,
                "annotations": {"coupling": {"semantic_coupling": 0.3}}}
        profile = profile_task(task)
        self.assertEqual(profile.semantic_coupling, 0.3)

    def test_derivation_report_origins(self):
        task = {"task_id": "t", "family": "allocation", "model": SIRL_MODEL,
                "annotations": {"coupling": {"semantic_coupling": 0.3}}}
        profile = profile_task(task)
        report = profile.annotations["profiling"]
        self.assertEqual(report["origin"]["resource_coupling"], "model")
        self.assertEqual(report["origin"]["semantic_coupling"], "supplied")
        self.assertIn("model_verification", report)

    def test_no_model_no_derivation_reports_null_with_note(self):
        task = {"task_id": "t", "family": "allocation", "spec": {}}
        profile = profile_task(task)
        report = profile.annotations["profiling"]
        self.assertEqual(report["origin"]["resource_coupling"], "null")

    def test_determinism_with_model(self):
        task = {"task_id": "t", "family": "allocation", "model": SIRL_MODEL}
        self.assertEqual(profile_task(task).to_dict(),
                         profile_task(task).to_dict())


class TestPendingStaging(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)

    def test_stage_and_pending(self):
        rec = self.make_record(execution_id="ex_p1", task_id="t1")
        self.bank.stage_pending(rec)
        pending = self.bank.pending()
        self.assertEqual([p.execution_id for p in pending], ["ex_p1"])
        self.assertEqual(self.bank.count(), 0)  # not recorded

    def test_clear_after_record(self):
        rec = self.make_record(execution_id="ex_p2")
        self.bank.stage_pending(rec)
        self.bank.append(rec)
        self.bank.clear_pending("ex_p2")
        self.assertEqual(self.bank.pending(), [])
        self.assertEqual(self.bank.count(), 1)

    def test_get_pending_verbatim(self):
        rec = self.make_record(execution_id="ex_p3", gap=0.2)
        self.bank.stage_pending(rec)
        staged = self.bank.get_pending("ex_p3")
        self.assertEqual(staged, rec)

    def test_cross_process_persistence(self):
        from or_harness.core.storage import Store
        self.bank.stage_pending(self.make_record(execution_id="ex_p4"))
        store2 = Store(self.home)
        try:
            bank2 = ExperienceBank(store2)
            self.assertEqual([p.execution_id for p in bank2.pending()], ["ex_p4"])
        finally:
            store2.close()


class TestCrossExecutionC4(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.catalog = load_catalog()

    def failed_record(self, execution_id, solver, error="security policy: blocked import subprocess"):
        rec = self.make_record(execution_id=execution_id, task_id="t1",
                               feasible=False, status="error",
                               solver={"name": solver, "code_hash": "x"})
        rec.failures = [FailureRecord(attempt=1, error=error)]
        return rec

    def test_recovery_chain_fires(self):
        failed = self.failed_record("ex_f1", "pulp")
        success = self.make_record(execution_id="ex_s1", task_id="t1",
                                   solver={"name": "ortools", "code_hash": "y"})
        hints = check_triggers(success, self.stats, self.catalog,
                               prior_failures=[failed])
        c4 = [h for h in hints if h.criterion == "C4"]
        self.assertTrue(c4)
        self.assertEqual(c4[0].evidence["kind"], "cross_execution_recovery")
        self.assertEqual(c4[0].evidence["failed"]["solver"], "pulp")
        self.assertEqual(c4[0].evidence["recovered_by"]["solver"], "ortools")

    def test_same_solver_retry_not_a_chain(self):
        failed = self.failed_record("ex_f2", "pulp")
        success = self.make_record(execution_id="ex_s2", task_id="t1",
                                   solver={"name": "pulp", "code_hash": "y"})
        hints = check_triggers(success, self.stats, self.catalog,
                               prior_failures=[failed])
        self.assertFalse(any(h.criterion == "C4" for h in hints))

    def test_failure_classification(self):
        env = self.failed_record("ex_c1", "pulp")
        model_bug = self.failed_record("ex_c2", "highs",
                                       error="ValueError: shapes not aligned")
        self.assertEqual(classify_failure(env), "environment")
        self.assertEqual(classify_failure(model_bug), "model")


class TestSolverAdvisories(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)

    def test_only_environment_failures_listed(self):
        env_fail = self.make_record(
            execution_id="ex_a", task_id="t1", feasible=False, status="error",
            solver={"name": "pulp", "code_hash": "x"},
            failures=[FailureRecord(1, "security policy: blocked import subprocess")])
        model_fail = self.make_record(
            execution_id="ex_b", task_id="t2", feasible=False, status="error",
            solver={"name": "highs", "code_hash": "y"},
            failures=[FailureRecord(1, "TypeError: unsupported operand")])
        ok = self.make_record(execution_id="ex_c", task_id="t3",
                              solver={"name": "ortools", "code_hash": "z"})
        for rec in (env_fail, model_fail, ok):
            self.bank.append(rec)
        advisories = solver_advisories(self.bank)
        self.assertEqual(len(advisories), 1)  # pulp only
        self.assertEqual(advisories[0]["solver"], "pulp")
        self.assertEqual(advisories[0]["environment_failures"], 1)
        self.assertEqual(advisories[0]["error_classes"], ["environment"])

    def test_empty_when_no_failures(self):
        self.bank.append(self.make_record(execution_id="ex_ok"))
        self.assertEqual(solver_advisories(self.bank), [])


class _StubExecutor:
    """Executes nothing: returns a pre-built evidence record."""

    def __init__(self, record):
        self._record = record

    def execute(self, code_path, workspace, *, solver, task_id, strategy_id,
                profile, verification_level="basic", code_hash=None):
        return self._record


class TestEvidenceKnowledgeSemantics(HarnessTestCase):
    """The two memory layers: Evidence records ACTUAL facts (including the
    CIR snapshot of the task actually solved); Knowledge entries inherit the
    catalog vocabulary (strategy_type/actions) without overwriting
    harness-supplied values."""

    @staticmethod
    def _cir():
        return {"entities": [{"name": "R1", "kind": "resource"}],
                "decisions": [{"name": "x"}],
                "constraints": [], "relations": [], "coupling_groups": []}

    def test_execute_captures_cir_snapshot(self):
        rec = self.make_record(execution_id="ex_cir1", task_id="t_cir",
                               strategy_id="S01")
        h = ORHarness(home=self.home, executor=_StubExecutor(rec))
        try:
            task = {"task_id": "t_cir", "family": "allocation",
                    "coupling": self._cir(), "spec": {}}
            out = h.execute(task, "S01", "solve.py", self.home, solver="highs")
            self.assertEqual(out.cir_snapshot, task["coupling"])
            # The snapshot persists through the Evidence Bank.
            h.record(out)
            stored = h.bank.get("ex_cir1")
            self.assertEqual(stored.cir_snapshot, task["coupling"])
        finally:
            h.close()

    def test_execute_without_cir_leaves_snapshot_none(self):
        rec = self.make_record(execution_id="ex_cir2", task_id="t_nocir",
                               strategy_id="S01")
        h = ORHarness(home=self.home, executor=_StubExecutor(rec))
        try:
            task = {"task_id": "t_nocir", "family": "allocation", "spec": {}}
            out = h.execute(task, "S01", "solve.py", self.home, solver="highs")
            self.assertIsNone(out.cir_snapshot)
        finally:
            h.close()

    def test_induce_inherits_catalog_vocabulary(self):
        h = ORHarness(home=self.home)
        try:
            h.catalog["S01"] = Strategy(
                strategy_id="S01", name="decompose",
                strategy_type="decomposition",
                actions=["find bottleneck", "decompose locals"])
            for i in range(2):
                h.bank.append(self.make_record(
                    execution_id=f"ex_enc{i}", task_id=f"te{i}",
                    strategy_id="S01", gap=0.05))
            result = h.induce(strategy_id="S01")
            entry = h.sbank.get(result["results"][0]["created"])
            self.assertEqual(entry.strategy_type, "decomposition")
            self.assertEqual(entry.actions,
                             ["find bottleneck", "decompose locals"])
            self.assertIsNone(entry.principle)  # v1 leaves principle empty
        finally:
            h.close()

    def test_rebuild_enriches_entries(self):
        h = ORHarness(home=self.home)
        try:
            h.catalog["S01"] = Strategy(strategy_id="S01", name="monolithic",
                                        strategy_type="modeling")
            for i in range(2):
                h.bank.append(self.make_record(
                    execution_id=f"ex_rb{i}", task_id=f"tr{i}",
                    strategy_id="S01", gap=0.05))
            result = h.induce(rebuild=True)
            self.assertEqual(result["rebuilt"], 1)
            entry = h.sbank.get(result["entry_ids"][0])
            self.assertEqual(entry.strategy_type, "modeling")
        finally:
            h.close()

    def test_enrichment_never_overwrites_harness_values(self):
        h = ORHarness(home=self.home)
        try:
            h.catalog["S01"] = Strategy(strategy_id="S01", name="x",
                                        strategy_type="modeling",
                                        actions=["catalog-action"])
            entry = StrategicEntry(
                entry_id="se_keep", strategy_id="S01",
                pattern={"scope_level": "L1", "predicates": {}},
                strategy_type="execution", actions=["harness-custom"])
            h.sbank.add(entry)
            h._enrich_entry("se_keep")
            kept = h.sbank.get("se_keep")
            self.assertEqual(kept.strategy_type, "execution")
            self.assertEqual(kept.actions, ["harness-custom"])
        finally:
            h.close()


class TestRetentionMarking(HarnessTestCase):
    """Representative-evidence retention marking at record time: lightweight
    origin metadata, not referential integrity."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def tearDown(self):
        self.h.close()

    def test_plain_success_unmarked(self):
        self.h.record(self.make_record(execution_id="ex_rt0", task_id="trt0"))
        self.assertIsNone(self.h.bank.get("ex_rt0").retention_reason)

    def test_boundary_failure_recovery_high_cost_marked(self):
        rec = self.make_record(execution_id="ex_rt1", task_id="trt1",
                               feasible=False, status="error")
        rec.failures = [FailureRecord(attempt=1, error="boom",
                                      recovery_action="switch solver")]
        self.h.record(rec)
        self.assertEqual(
            self.h.bank.get("ex_rt1").retention_reason,
            "boundary_outcome,failure_recovery,high_cost")

    def test_recovery_chain_marked(self):
        failed = self.make_record(execution_id="ex_rtf", task_id="trt2",
                                  feasible=False, status="error")
        failed.failures = [FailureRecord(attempt=1, error="x")]
        self.h.bank.stage_pending(failed)
        ok = self.make_record(execution_id="ex_rts", task_id="trt2")
        self.h.record(ok)
        self.assertEqual(self.h.bank.get("ex_rts").retention_reason,
                         "recovery_chain")

    def test_retries_alone_marked_high_cost(self):
        rec = self.make_record(execution_id="ex_rt4", task_id="trt4",
                               cost=CostVector(llm_tokens=100, retries=2))
        self.h.record(rec)
        self.assertEqual(self.h.bank.get("ex_rt4").retention_reason,
                         "high_cost")

    def test_harness_override_wins(self):
        rec = self.make_record(execution_id="ex_rt5", task_id="trt5")
        self.h.record(rec, retain_reason="contrast")
        self.assertEqual(self.h.bank.get("ex_rt5").retention_reason,
                         "contrast")


if __name__ == "__main__":
    unittest.main()
