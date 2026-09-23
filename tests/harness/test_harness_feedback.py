"""Tests for the harness-feedback fixes: model representation, coupling
derivation chain, cross-check warnings, pending staging, cross-execution
intervention recovery,
and solver advisories. The SIRL_026 scenario (independent resource
constraints mis-supplied as rc=0.8) is reproduced end to end."""
import json
import textwrap
import unittest

from helpers import HarnessTestCase, MEASURED_ALL

from or_harness.api import ORHarness
from or_harness.core.schema import CostVector, FailureRecord, StrategicEntry
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
    def test_model_is_diagnostic_not_the_key(self):
        """The model is a POST-strategy artifact: its coupling is reported
        as a diagnostic, but the structural KEY comes from pre-strategy
        inputs only (here: the supplied annotations)."""
        task = {
            "task_id": "SIRL_026", "family": "allocation",
            "model": SIRL_MODEL,
            "annotations": {"coupling": {"resource_coupling": 0.8}},
        }
        profile = profile_task(task)
        # The supplied value IS the key; the model does not move it.
        self.assertEqual(profile.resource_coupling, 0.8)
        report = profile.annotations["profiling"]
        self.assertEqual(report["origin"]["resource_coupling"], "supplied")
        # The model's own reading is still visible as a diagnostic.
        self.assertIn("model_coupling", report)
        self.assertEqual(report["model_coupling"]["resource_coupling"], 0.0)
        # The diagnostic differs from the key — visibly, not silently.
        self.assertNotEqual(profile.resource_coupling,
                            report["model_coupling"]["resource_coupling"])

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
        # No CIR and no spec: the structural dims stay unknown — a model
        # never becomes the identity source.
        self.assertEqual(report["origin"]["resource_coupling"], "null")
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

    def test_spec_still_defines_the_key(self):
        """A PRE-strategy structured spec IS an identity source."""
        task = {"task_id": "t", "family": "routing",
                "spec": {"time_periods": 24}}
        profile = profile_task(task)
        report = profile.annotations["profiling"]
        self.assertEqual(report["origin"]["temporal_coupling"], "spec")
        self.assertIsNotNone(profile.temporal_coupling)


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


class TestCrossExecutionRecovery(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.stats = ConditionalStats(self.bank)

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
        hints = check_triggers(success, self.stats,
                               prior_failures=[failed])
        c4 = [h for h in hints if h.pattern == "intervention_recovery"]
        self.assertTrue(c4)
        self.assertEqual(c4[0].evidence["kind"], "cross_execution_recovery")
        self.assertEqual(c4[0].evidence["failed"]["solver"], "pulp")
        self.assertEqual(c4[0].evidence["recovered_by"]["solver"], "ortools")

    def test_same_solver_retry_not_a_chain(self):
        failed = self.failed_record("ex_f2", "pulp")
        success = self.make_record(execution_id="ex_s2", task_id="t1",
                                   solver={"name": "pulp", "code_hash": "y"})
        hints = check_triggers(success, self.stats,
                               prior_failures=[failed])
        self.assertFalse(any(h.pattern == "intervention_recovery"
                             for h in hints))

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
    CIR snapshot of the task actually solved); Knowledge entries carry only
    what the harness (or a migration) put on them — the framework fills in no
    method vocabulary from any directory."""

    @staticmethod
    def _cir():
        return {"entities": [{"name": "R1", "kind": "resource"}],
                "decisions": [{"name": "x"}],
                "constraints": [], "relations": [], "coupling_groups": []}

    def test_execute_captures_cir_snapshot(self):
        """The snapshot is the PARSED CIR, not the caller's raw payload.

        Storing the raw payload made the record a false-positive signal: a
        nested ``{"cir": {...}}`` froze verbatim, so the record looked like
        it carried a structure while every consumer parsed zero entities.
        """
        from or_harness.core.coupling import CouplingAwareIR
        rec = self.make_record(execution_id="ex_cir1", task_id="t_cir",
                               strategy_id="S01")
        h = ORHarness(home=self.home, executor=_StubExecutor(rec))
        try:
            task = {"task_id": "t_cir", "family": "allocation",
                    "coupling": self._cir(), "spec": {}}
            out = h.execute(task, "S01", "solve.py", self.home, solver="highs")
            expected = CouplingAwareIR.from_dict(self._cir()).to_dict()
            self.assertEqual(out.cir_snapshot, expected)
            # The snapshot persists through the Evidence Bank.
            h.record(out)
            stored = h.bank.get("ex_cir1")
            self.assertEqual(stored.cir_snapshot, expected)
        finally:
            h.close()

    def test_execute_refuses_a_malformed_cir(self):
        """A malformed CIR is a precondition failure, not a silent empty
        structure — execute must not run a solve under a CIR nobody parsed."""
        from or_harness.core.coupling import CIRFormatError
        rec = self.make_record(execution_id="ex_cir3", task_id="t_cir3",
                               strategy_id="S01")
        h = ORHarness(home=self.home, executor=_StubExecutor(rec))
        try:
            task = {"task_id": "t_cir3", "family": "allocation",
                    "coupling": {"cir": self._cir()}, "spec": {}}
            with self.assertRaises(CIRFormatError) as caught:
                h.execute(task, "S01", "solve.py", self.home, solver="highs")
            self.assertIn("NESTED", caught.exception.hint)
            # Nothing was staged: the refusal happens before the solve.
            self.assertEqual(h.bank.pending(), [])
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

    def test_induce_does_not_invent_method_vocabulary(self):
        """Induction forms a CLAIM; it does not describe the method.

        There is no directory to copy strategy_type/actions/fallback from, so
        an induced entry reports them as unrecorded — which is what the memory
        really knows.
        """
        h = ORHarness(home=self.home)
        try:
            for i in range(2):
                h.bank.append(self.make_record(
                    execution_id=f"ex_enc{i}", task_id=f"te{i}",
                    strategy_id="S01", gap=0.05))
            result = h.induce(strategy_id="S01")
            entry = h.sbank.get(result["results"][0]["created"])
            self.assertIsNone(entry.strategy_type)
            self.assertEqual(entry.actions, [])
            self.assertIsNone(entry.fallback_strategy_id)
        finally:
            h.close()

    def test_harness_supplied_vocabulary_survives_induction(self):
        """A harness that DOES record the method's content keeps it."""
        h = ORHarness(home=self.home)
        try:
            entry = StrategicEntry(
                entry_id="se_keep", strategy_id="S01",
                pattern={"predicates": {}},
                strategy_type="execution", actions=["harness-custom"],
                fallback_strategy_id="custom:fallback")
            h.sbank.add(entry)
            for i in range(2):
                h.bank.append(self.make_record(
                    execution_id=f"ex_keep{i}", task_id=f"tk{i}",
                    strategy_id="S01", gap=0.05))
            h.induce(strategy_id="S01")
            kept = h.sbank.get("se_keep")
            self.assertEqual(kept.strategy_type, "execution")
            self.assertEqual(kept.actions, ["harness-custom"])
            self.assertEqual(kept.fallback_strategy_id, "custom:fallback")
        finally:
            h.close()

    def test_rebuild_does_not_invent_method_vocabulary(self):
        h = ORHarness(home=self.home)
        try:
            for i in range(2):
                h.bank.append(self.make_record(
                    execution_id=f"ex_rb{i}", task_id=f"tr{i}",
                    strategy_id="S01", gap=0.05))
            result = h.induce(rebuild=True)
            self.assertEqual(result["rebuilt"], 1)
            entry = h.sbank.get(result["entry_ids"][0])
            self.assertIsNone(entry.strategy_type)
        finally:
            h.close()


class TestCostCompletenessSignal(HarnessTestCase):
    """Recording never blocks, but it must SAY what the cost data cannot
    support: a dimension left unmeasured cannot back a strategic-entry cost
    claim or a cost prediction."""

    def test_missing_dimensions_are_reported(self):
        h = ORHarness(home=self.home)
        try:
            rec = self.make_record(
                execution_id="ex_cc1", task_id="tcc1",
                cost=CostVector(solver_runtime_s=1.0, latency_s=0.5,
                                measured={"solver_runtime_s", "latency_s"}))
            out = h.record(rec)
            block = out["cost_completeness"]
            self.assertEqual(set(block["missing"]),
                             {"llm_tokens", "tool_calls", "retries"})
            self.assertIn("UNKNOWN", block["note"])
            self.assertIn("--override", block["note"])
            # The fact is still recorded — a warning, not a refusal.
            self.assertTrue(out["recorded"])
            self.assertIsNotNone(h.bank.get("ex_cc1"))
        finally:
            h.close()

    def test_no_signal_when_everything_is_measured(self):
        h = ORHarness(home=self.home)
        try:
            rec = self.make_record(
                execution_id="ex_cc2", task_id="tcc2",
                cost=CostVector(llm_tokens=100.0, tool_calls=2.0,
                                solver_runtime_s=1.0, retries=0.0,
                                latency_s=0.5),
                cost_measured=MEASURED_ALL)
            out = h.record(rec)
            self.assertNotIn("cost_completeness", out)
        finally:
            h.close()

    def test_lower_bound_is_reported_alongside_the_gap(self):
        h = ORHarness(home=self.home)
        try:
            rec = self.make_record(
                execution_id="ex_cc3", task_id="tcc3",
                cost=CostVector(solver_runtime_s=1.0,
                                measured={"solver_runtime_s"}))
            rec.execution_features["tool_calls_lower_bound"] = 1
            out = h.record(rec)
            self.assertEqual(out["cost_completeness"]["lower_bounds"],
                             {"tool_calls_lower_bound": 1})
        finally:
            h.close()

    def test_declared_tool_calls_below_the_floor_is_refused(self):
        """The record itself refutes the number: the executor demonstrably
        ran the script, so tool_calls cannot be 0."""
        from or_harness.core.storage import StorageError
        h = ORHarness(home=self.home)
        try:
            rec = self.make_record(
                execution_id="ex_cc4", task_id="tcc4",
                cost=CostVector(solver_runtime_s=1.0,
                                measured={"solver_runtime_s"}))
            rec.execution_features["tool_calls_lower_bound"] = 1
            h.bank.append(rec)
            with self.assertRaises(StorageError) as caught:
                h.bank.update_cost("ex_cc4", tool_calls=0.0)
            self.assertIn("lower bound", str(caught.exception))
            # A declaration AT the floor is fine.
            h.bank.update_cost("ex_cc4", tool_calls=1.0)
            self.assertEqual(h.bank.get("ex_cc4").cost.tool_calls, 1.0)
        finally:
            h.close()


class TestProvableRetries(HarnessTestCase):
    """retries=0 is claimed only when the framework can PROVE there was
    nothing to retry — otherwise the count stays unknown."""

    def _script(self, name="solve_ret.py"):
        from pathlib import Path
        path = Path(self.home) / name
        path.write_text(
            "import json\n"
            "json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "           'objective_bound': 1.0, 'mip_gap': 0.0},\n"
            "          open('result.json', 'w'))\n", encoding="utf-8")
        return str(path)

    def _task(self, tid):
        return {"task_id": tid, "family": "routing", "spec": {}}

    def test_first_attempt_has_a_proven_zero(self):
        h = ORHarness(home=self.home)
        try:
            rec = h.execute(self._task("tr1"), "S01", self._script(), self.home,
                            solver="highs", episode_id="ep1")
            self.assertEqual(rec.cost.retries, 0.0)
            self.assertIn("retries", rec.cost.measured_dims())
            self.assertIn("no earlier execute_strategy attempt",
                          rec.execution_features["retries_proof"])
        finally:
            h.close()

    def test_second_attempt_of_the_same_strategy_is_unknown(self):
        """A second attempt of the same (task, episode, strategy) may or may
        not be a retry — the framework must not guess, so the count stays
        unmeasured until the harness declares it."""
        h = ORHarness(home=self.home)
        try:
            first = h.execute(self._task("tr2"), "S01", self._script("a.py"),
                              self.home, solver="highs", episode_id="ep2")
            h.record(first)
            second = h.execute(self._task("tr2"), "S01", self._script("b.py"),
                               self.home, solver="highs", episode_id="ep2")
            self.assertNotIn("retries", second.cost.measured_dims())
            self.assertNotIn("retries_proof", second.execution_features)
            # The harness declares the relationship explicitly.
            h.record(second, override={"retries": 1.0})
            self.assertEqual(h.bank.get(second.execution_id).cost.retries, 1.0)
            self.assertIn("retries",
                          h.bank.get(second.execution_id).cost.measured_dims())
        finally:
            h.close()

    def test_a_different_strategy_is_not_a_prior_attempt(self):
        h = ORHarness(home=self.home)
        try:
            first = h.execute(self._task("tr3"), "S01", self._script("c.py"),
                              self.home, solver="highs", episode_id="ep3")
            h.record(first)
            other = h.execute(self._task("tr3"), "S02", self._script("d.py"),
                              self.home, solver="highs", episode_id="ep3")
            self.assertEqual(other.cost.retries, 0.0)
            self.assertIn("retries", other.cost.measured_dims())
        finally:
            h.close()

    def test_a_different_episode_is_not_a_prior_attempt(self):
        h = ORHarness(home=self.home)
        try:
            first = h.execute(self._task("tr4"), "S01", self._script("e.py"),
                              self.home, solver="highs", episode_id="epA")
            h.record(first)
            other = h.execute(self._task("tr4"), "S01", self._script("f.py"),
                              self.home, solver="highs", episode_id="epB")
            self.assertEqual(other.cost.retries, 0.0)
            self.assertIn("retries", other.cost.measured_dims())
        finally:
            h.close()


class TestRetentionMarking(HarnessTestCase):
    """Explicit-only retention marks: an explicit param wins; otherwise the
    record's existing mark is preserved. No automatic marking."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def tearDown(self):
        self.h.close()

    def test_no_auto_marking(self):
        rec = self.make_record(execution_id="ex_rt0", task_id="trt0",
                               feasible=False, status="error")
        rec.failures = [FailureRecord(attempt=1, error="boom",
                                      recovery_action="switch solver")]
        self.h.record(rec)
        self.assertIsNone(self.h.bank.get("ex_rt0").retention_reason)

    def test_existing_mark_preserved_without_explicit_param(self):
        rec = self.make_record(execution_id="ex_rt1", task_id="trt1")
        rec.retention_reason = "contrast"
        self.h.record(rec)
        self.assertEqual(self.h.bank.get("ex_rt1").retention_reason,
                         "contrast")

    def test_explicit_param_wins_over_existing_mark(self):
        rec = self.make_record(execution_id="ex_rt2", task_id="trt2")
        rec.retention_reason = "old"
        self.h.record(rec, retain_reason="new-reason")
        self.assertEqual(self.h.bank.get("ex_rt2").retention_reason,
                         "new-reason")

    def test_blank_explicit_param_ignored(self):
        rec = self.make_record(execution_id="ex_rt3", task_id="trt3")
        rec.retention_reason = "keep"
        self.h.record(rec, retain_reason="   ")
        self.assertEqual(self.h.bank.get("ex_rt3").retention_reason, "keep")


if __name__ == "__main__":
    unittest.main()
