"""Do the recorded numbers have the standing to be used as TRUTHS?

A cost dimension can pass the storage mask (``measured``) and still not be
usable as a measured fact. This file pins the ONE rule
(:mod:`or_harness.world_model.cost_eligibility`) and the places it reaches:

1. **A declared estimate is not a truth.** ``source="agent_estimate"``
   keeps the value visible on the record and in the views, but it never
   becomes a calibration actual, never enters a learning evidence mean, and
   never manufactures cost feedback.
2. **A single-side token count is a lower bound.** ``basis ==
   completion_only`` cannot stand in for a complete ``llm_tokens`` total.
3. **A trusted explicit zero is a real measurement.** The rule targets the
   SOURCE, not the magnitude: 0 from the provider is a true zero.
4. **Unknown is never zero.** A dimension nobody measured produces no
   actual, no mean and no claim — never a fabricated 0.
5. **A late host report can backfill through the CLI** (``amend-cost
   --usage-file``), the channel the advice printed by ``record``/``induce``
   actually points at.
6. **A host report records ``provider_usage``**, not whatever source the
   caller's default was.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    PredictionSnapshot,
    ProblemProfile,
)
from or_harness.world_model.usage import (  # noqa: E402
    dedupe_host_reports,
    host_usage_cost_vector,
    usage_cost_vector,
)
from or_harness.world_model.cost_eligibility import (  # noqa: E402
    ESTIMATE,
    MEASURED,
    PARTIAL,
    UNKNOWN,
    cost_eligibility,
)


def _profile(problem_id="t1"):
    return ProblemProfile(problem_id=problem_id, family="routing")


def _record(tokens=1000.0, source=None, basis=None, measured=None,
            execution_id="ex_1"):
    provenance = {}
    if source is not None or basis is not None:
        entry = {}
        if source is not None:
            entry["source"] = source
        if basis is not None:
            entry["basis"] = basis
        provenance["llm_tokens"] = entry
    return ExecutionRecord(
        execution_id=execution_id, task_id="t1", strategy_id="S01",
        profile_snapshot=_profile(),
        quality={"feasible": True},
        cost=CostVector(llm_tokens=tokens,
                        measured=(set(measured) if measured is not None
                                  else {"llm_tokens"})),
        execution_features=({"cost_provenance": provenance}
                            if provenance else {}),
    )


# ---------------------------------------------------------------------------
# 1. the pure rule
# ---------------------------------------------------------------------------


class TestTheRule(unittest.TestCase):

    def test_a_provider_report_is_measured(self):
        self.assertEqual(
            cost_eligibility(_record(source="provider_usage"), "llm_tokens"),
            MEASURED)

    def test_an_observed_count_is_measured(self):
        self.assertEqual(
            cost_eligibility(_record(source="agent_observed"), "llm_tokens"),
            MEASURED)

    def test_a_declared_estimate_is_not_a_truth(self):
        self.assertEqual(
            cost_eligibility(_record(source="agent_estimate"), "llm_tokens"),
            ESTIMATE)

    def test_an_unmarked_provider_measurement_is_measured(self):
        # A record the executor itself stamped (no declaration): the
        # framework measured it, so it is a fact.
        self.assertEqual(
            cost_eligibility(_record(source=None), "llm_tokens"), MEASURED)

    def test_a_single_side_token_figure_is_partial(self):
        self.assertEqual(
            cost_eligibility(
                _record(source="provider_usage",
                        basis="completion_only"), "llm_tokens"), PARTIAL)
        self.assertEqual(
            cost_eligibility(
                _record(source="provider_usage",
                        basis="prompt_only"), "llm_tokens"), PARTIAL)

    def test_an_unmeasured_dimension_is_unknown(self):
        rec = _record(measured=set())
        self.assertEqual(cost_eligibility(rec, "llm_tokens"), UNKNOWN)

    def test_a_trusted_explicit_zero_is_measured(self):
        # The rule targets the SOURCE, not the magnitude: a provider that
        # really reported 0 measured a true zero.
        rec = _record(tokens=0.0, source="provider_usage")
        self.assertEqual(cost_eligibility(rec, "llm_tokens"), MEASURED)

    def test_an_estimated_zero_is_still_not_a_truth(self):
        rec = _record(tokens=0.0, source="agent_estimate")
        self.assertEqual(cost_eligibility(rec, "llm_tokens"), ESTIMATE)


# ---------------------------------------------------------------------------
# 2. calibration: an estimate is not an actual
# ---------------------------------------------------------------------------


class TestCalibration(HarnessTestCase):

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def test_a_declared_token_value_is_excluded_from_the_real_total(self):
        rec = self.make_record(task_id="t1", strategy_id="S01",
                               cost=CostVector(llm_tokens=20000.0,
                                               solver_runtime_s=1.0,
                                               measured={"llm_tokens",
                                                         "solver_runtime_s"}))
        rec.execution_features["cost_provenance"] = {
            "llm_tokens": {"source": "agent_estimate"}}
        self.h.bank.append(rec)
        from or_harness.world_model.episode_closeout import _aggregate_costs
        aggregate = _aggregate_costs([rec.cost], [rec])
        # The declaration is dropped from the total...
        self.assertIsNone(aggregate["llm_tokens"]["total"])
        self.assertEqual(aggregate["llm_tokens"]["n_measured"], 0)
        self.assertEqual(aggregate["llm_tokens"]["excluded"], 1)

    def test_a_provider_measurement_is_the_total(self):
        rec = self.make_record(task_id="t1", strategy_id="S01",
                               cost=CostVector(llm_tokens=1500.0,
                                               measured={"llm_tokens"}))
        rec.execution_features["cost_provenance"] = {
            "llm_tokens": {"source": "provider_usage"}}
        self.h.bank.append(rec)
        from or_harness.world_model.episode_closeout import _aggregate_costs
        aggregate = _aggregate_costs([rec.cost], [rec])
        self.assertEqual(aggregate["llm_tokens"]["total"], 1500.0)
        self.assertNotIn("excluded", aggregate["llm_tokens"])


class TestTheDefaultIsADeclaration(HarnessTestCase):
    """A bare hand-typed cost number must not become a calibration truth.

    The safe default and the honest default are the same one: a number
    typed with no stated source is an ``agent_estimate``. The caller opts
    IN to a trusted source when the number really came from a report.
    """

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def test_update_cost_defaults_to_a_declaration(self):
        rec = self.make_record(task_id="t1", strategy_id="S01")
        self.h.bank.append(rec)
        stored = self.h.bank.update_cost(rec.execution_id, llm_tokens=5000.0)
        self.assertEqual(
            stored.execution_features["cost_provenance"]["llm_tokens"]
            ["source"], "agent_estimate")
        self.assertEqual(cost_eligibility(stored, "llm_tokens"), ESTIMATE)

    def test_record_defaults_to_a_declaration(self):
        rec = self.make_record(
            task_id="t1", strategy_id="S01",
            cost=CostVector(llm_tokens=0.0,
                            measured={"latency_s", "solver_runtime_s"}),
            cost_measured=("latency_s", "solver_runtime_s"))
        outcome = self.h.record(rec, override={"llm_tokens": 5000.0})
        stored = self.h.bank.get(outcome["execution_id"])
        self.assertEqual(
            stored.execution_features["cost_provenance"]["llm_tokens"]
            ["source"], "agent_estimate")
        self.assertEqual(cost_eligibility(stored, "llm_tokens"), ESTIMATE)

    def test_opting_in_to_a_trusted_source_makes_it_measured(self):
        rec = self.make_record(task_id="t1", strategy_id="S01")
        self.h.bank.append(rec)
        stored = self.h.bank.update_cost(
            rec.execution_id, llm_tokens=5000.0, source="agent_observed")
        self.assertEqual(cost_eligibility(stored, "llm_tokens"), MEASURED)


# ---------------------------------------------------------------------------
# 3. learning evidence: an estimate does not enter the mean
# ---------------------------------------------------------------------------


class TestConditionalStats(HarnessTestCase):

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def _cell(self, rec):
        from or_harness.strategy.stats import ConditionalStats
        return ConditionalStats(self.h.bank).cell(rec.group_l1, "S01")

    def test_a_declared_estimate_is_excluded_from_the_mean(self):
        rec = self.make_record(task_id="t1", strategy_id="S01",
                                 cost=CostVector(llm_tokens=20000.0,
                                                 solver_runtime_s=1.0,
                                                 measured={"llm_tokens",
                                                           "solver_runtime_s"}))
        rec.execution_features["cost_provenance"] = {
            "llm_tokens": {"source": "agent_estimate"}}
        self.h.bank.append(rec)
        cell = self._cell(rec)
        self.assertEqual(cell.n_measured.get("llm_tokens", 0), 0)
        self.assertIsNone(cell.measured_cost("llm_tokens"))
        # The measured dimension is unaffected.
        self.assertEqual(cell.n_measured.get("solver_runtime_s"), 1)

    def test_a_provider_measurement_enters_the_mean(self):
        rec = self.make_record(task_id="t1", strategy_id="S01",
                                 cost=CostVector(llm_tokens=1500.0,
                                                 measured={"llm_tokens"}))
        rec.execution_features["cost_provenance"] = {
            "llm_tokens": {"source": "provider_usage"}}
        self.h.bank.append(rec)
        cell = self._cell(rec)
        self.assertEqual(cell.n_measured.get("llm_tokens"), 1)
        self.assertAlmostEqual(cell.measured_cost("llm_tokens"), 1500.0)


# ---------------------------------------------------------------------------
# 4. feedback: an estimate never manufactures an error
# ---------------------------------------------------------------------------


class TestFeedback(HarnessTestCase):

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def _snapshot(self):
        return PredictionSnapshot(
            strategy_id="S01",
            expected_cost=CostVector(llm_tokens=100.0,
                                     measured={"llm_tokens"}),
            source="entry", support_n=2, support_per_dim={"llm_tokens": 2})

    def _seed(self):
        rec = self.make_record(
            task_id="t1", strategy_id="S01",
            cost=CostVector(llm_tokens=0.0, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0),
            cost_measured=("tool_calls", "solver_runtime_s", "retries",
                           "latency_s"))
        return self.h.record(rec, prediction=self._snapshot())

    def test_an_estimated_backfill_does_not_manufacture_feedback(self):
        outcome = self._seed()
        self.h.bank.update_cost(outcome["execution_id"], llm_tokens=20000.0,
                                source="agent_estimate")
        stored = self.h.bank.get(outcome["execution_id"])
        feedback = stored.execution_features.get("cost_feedback")
        if feedback is not None:
            self.assertNotIn("llm_tokens", feedback.get("per_dimension", {}))
            self.assertIn("llm_tokens", feedback.get("excluded_dims", {}))
        # The value itself stays on the record, visible, not silently dropped.
        self.assertEqual(stored.cost.llm_tokens, 20000.0)

    def test_a_provider_backfill_produces_feedback(self):
        outcome = self._seed()
        self.h.bank.update_cost(outcome["execution_id"], llm_tokens=1000.0,
                                source="provider_usage")
        stored = self.h.bank.get(outcome["execution_id"])
        feedback = stored.execution_features["cost_feedback"]
        self.assertAlmostEqual(
            feedback["per_dimension"]["llm_tokens"]["actual"], 1000.0)


# ---------------------------------------------------------------------------
# 5. the budget view: a usage-less call keeps the verdict honest
# ---------------------------------------------------------------------------


class TestBudgetCompleteness(HarnessTestCase):

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def test_an_action_with_unknown_cost_makes_the_budget_unconfirmed(self):
        action = self.h.begin_action(
            "model", {"task_id": "t1", "family": "routing"}, "ep1")
        self.h.actions.end_action(action["action_id"],
                                  outcome={"note": "no usage reported"},
                                  cost=None)
        view = self.h.budget_view("t1", "ep1")
        entry = [c for c in view["consumption"]["action_costs"]
                 if c["action_id"] == action["action_id"]][0]
        self.assertIsNone(entry["cost"])
        view2 = self.h.budget_view("t1", "ep1", budget={"llm_tokens": 10.0})
        self.assertEqual(view2["status"], "unconfirmed")


# ---------------------------------------------------------------------------
# 6. the CLI late-backfill channel
# ---------------------------------------------------------------------------


class TestCliBackfillChannel(HarnessTestCase):

    def _solve(self, name):
        work = Path(self.home) / name
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 100.0,\n"
            "               'objective_bound': 100.0, 'runtime_seconds': 0.01},"
            " fh)\n", encoding="utf-8")
        return work

    def test_amend_cost_accepts_a_host_usage_report(self):
        from or_harness import cli as cli_module
        from or_harness.strategy.embedding_index import (
            LocalHashEmbeddingBackend,
        )
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        work = self._solve("late")
        record = h.execute({"task_id": "t1", "family": "routing", "spec": {}},
                           "S01", str(work / "solve.py"), str(work),
                           solver="highs", episode_id="ep1")
        h.record(record)
        usage_path = Path(self.home) / "late_usage.json"
        usage_path.write_text(json.dumps(
            {"prompt_tokens": 700, "completion_tokens": 300,
             "source": "hermes"}), encoding="utf-8")
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        args = type("Args", (), {
            "execution_id": record.execution_id, "override": None,
            "mode": "replace", "source": "agent_observed",
            "force": False, "usage_file": str(usage_path),
            "usage_source": None,
        })()
        code = cli_module.cmd_amend_cost(args)
        self.assertEqual(code, 0)
        stored = h.bank.get(record.execution_id)
        self.assertEqual(stored.cost.llm_tokens, 1000.0)
        provenance = stored.execution_features["cost_provenance"]["llm_tokens"]
        # A host report is a real observation, recorded as provider_usage.
        self.assertEqual(provenance["source"], "provider_usage")
        self.assertEqual(provenance["basis"], "provider_total")


# ---------------------------------------------------------------------------
# 7. repeated host events are counted once
# ---------------------------------------------------------------------------


class TestHostReportIdempotency(unittest.TestCase):

    def test_a_repeated_report_id_is_dropped_whole(self):
        report = {"report_id": "r1", "prompt_tokens": 100,
                  "completion_tokens": 50}
        kept, summary = dedupe_host_reports([report, dict(report)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(summary["dropped_reports"], 1)

    def test_a_repeated_call_id_is_dropped(self):
        report = {"calls": [
            {"id": "c1", "usage": {"prompt_tokens": 100,
                                   "completion_tokens": 50}}]}
        again = {"calls": [
            {"id": "c1", "usage": {"prompt_tokens": 100,
                                   "completion_tokens": 50}}]}
        kept, summary = dedupe_host_reports([report, again])
        calls = [c for r in kept for c in (r.get("calls") or [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary["dropped_calls"], 1)

    def test_a_late_report_replaces_instead_of_adding(self):
        first = {"calls": [{"id": "c1", "usage": {"prompt_tokens": 100,
                                                  "completion_tokens": 50}}]}
        late = {"late": True,
                "calls": [{"id": "c1", "usage": {"prompt_tokens": 100,
                                                 "completion_tokens": 80}}]}
        kept, summary = dedupe_host_reports([first, late])
        calls = [c for r in kept for c in (r.get("calls") or [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["usage"]["completion_tokens"], 80)
        self.assertEqual(summary["replaced_calls"], 1)

    def test_the_vector_sums_a_corrected_call_once(self):
        report = {"late": True, "calls": [
            {"id": "c1", "usage": {"prompt_tokens": 100,
                                   "completion_tokens": 80}}]}
        vector, breakdown = host_usage_cost_vector(report)
        self.assertEqual(vector.llm_tokens, 180.0)

    def test_an_uncorrected_duplicate_never_double_counts_via_replace(self):
        # No call ids: the report is still applied once (replace is
        # idempotent), so a replayed identical report cannot double-count.
        report = {"prompt_tokens": 100, "completion_tokens": 50}
        vector, _ = host_usage_cost_vector(report)
        self.assertEqual(vector.llm_tokens, 150.0)


if __name__ == "__main__":
    unittest.main()
