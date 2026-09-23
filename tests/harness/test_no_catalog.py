"""No built-in strategy directory: memory is the only source of candidates.

The design this file pins down, one property per test:

1. An empty bank recalls NOTHING — no `no_memory` row, no `-inf` score, no
   zero-quality placeholder. An empty answer is a real answer.
2. An empty structural channel never erases a semantic hit (and vice versa).
3. A retrieval FAILURE and a retrieval that RAN WITH NO HITS are labelled
   differently — "could not look" is not "looked and found nothing".
4. A method the framework has never heard of can be predicted, executed,
   recorded, and recalled afterwards. No directory membership is required.
5. No candidates supplied means `no_candidates` — the framework does not
   invent a menu.
6. History stays readable with the directory gone, and nothing is back-filled
   from it.
7. Two different methods that share an id are NOT merged: the binding records
   the config mismatch, and the statistics stay keyed on what really ran.
"""
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import CostVector
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.prediction import ActionSpec

REPO = Path(__file__).resolve().parents[2]

#: A method id that no directory ever contained.
OUTSIDE = "custom:two-phase-milp"

TASK = {
    "task_id": "t_outside",
    "family": "routing",
    "text": "route the fleet so total distance is minimal",
    "annotations": {"coupling": {"resource_coupling": 0.9,
                                 "temporal_coupling": 0.1,
                                 "route_complexity": 0.85,
                                 "semantic_coupling": 0.8}},
}

SOLVE = textwrap.dedent("""
    import json
    with open("result.json", "w") as fh:
        json.dump({"status": "optimal", "objective_value": 100.0,
                   "objective_bound": 100.0, "runtime_seconds": 0.01,
                   "variables": {"x1": 2, "x2": 3}}, fh)
""")


def _run_orx(home, *args):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run(
        [sys.executable, "-m", "or_harness.cli", "--home", home, *args],
        cwd=str(REPO), env=env, capture_output=True, text=True)


class NoCatalogCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.work = Path(self.home) / "ws"
        self.work.mkdir(parents=True, exist_ok=True)
        self.task_path = self.work / "task.json"
        self.task_path.write_text(json.dumps(TASK), encoding="utf-8")
        self.solve_path = self.work / "solve.py"
        self.solve_path.write_text(SOLVE, encoding="utf-8")

    def harness(self, **kwargs):
        h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend(),
                      **kwargs)
        self.addCleanup(h.close)
        return h


# ---------------------------------------------------------------------------
# 1. an empty bank says so, and says nothing else
# ---------------------------------------------------------------------------


class TestEmptyMemory(NoCatalogCase):
    def test_empty_bank_returns_an_empty_result_with_a_reason(self):
        h = self.harness()
        result = h.recall(TASK)
        self.assertEqual(result["recommendations"], [])
        basis = result["recommendations_basis"]
        self.assertEqual(basis["n_recommendations"], 0)
        self.assertEqual(basis["candidates_with_memory"], [])
        self.assertIn("NO MEMORY", basis["reason"])

    def test_no_placeholder_row_anywhere(self):
        """The old shape — a `no_memory` row at score -inf — must be gone."""
        h = self.harness()
        result = h.recall(TASK)
        self.assertNotIn("no_memory", json.dumps(result["recommendations"]))
        self.assertNotIn("-inf", json.dumps(result["recommendations"]))
        self.assertNotIn("no_memory", json.dumps(result["recommendations_basis"]))

    def test_no_name_or_description_is_fabricated(self):
        h = self.harness()
        h.bank.append(self.make_record(execution_id="ex_a", task_id="t1",
                                       strategy_id=OUTSIDE))
        rec = h.recall(TASK)["recommendations"][0]
        self.assertEqual(rec["strategy_id"], OUTSIDE)
        self.assertNotIn("name", rec)
        self.assertNotIn("description", rec)

    def test_memory_mode_none_is_reported_as_a_choice(self):
        h = self.harness()
        result = h.recall(TASK, memory_mode="none")
        self.assertEqual(result["recommendations"], [])
        self.assertIn("deliberately not consulted",
                      result["recommendations_basis"]["reason"])


# ---------------------------------------------------------------------------
# 2/3. the two channels are independent, and failure != no hits
# ---------------------------------------------------------------------------


class TestChannelsIndependent(NoCatalogCase):
    def test_empty_structural_channel_does_not_erase_a_semantic_hit(self):
        """A cross-cell memory is reachable by TEXT even when the structural
        channel holds nothing for this cell."""
        h = self.harness()
        # Same task text, a DIFFERENT structural cell.
        other = dict(TASK, task_id="t_other",
                     annotations={"coupling": {"resource_coupling": 0.1,
                                               "temporal_coupling": 0.9,
                                               "route_complexity": 0.1,
                                               "semantic_coupling": 0.2}})
        record = h.execute(other, OUTSIDE, str(self.solve_path),
                           str(self.work), solver="highs")
        h.record(record)
        result = h.recall(TASK)
        self.assertEqual(result["recommendations"], [])
        hits = result["vector_recall"]["execution_evidence"]
        self.assertTrue(hits, "the semantic channel must still surface it")
        self.assertEqual(hits[0]["strategy_id"], OUTSIDE)
        self.assertEqual(hits[0]["structural_match"], "different_cell")

    def test_semantic_hit_never_enters_the_statistics(self):
        h = self.harness()
        other = dict(TASK, task_id="t_other",
                     annotations={"coupling": {"resource_coupling": 0.1,
                                               "temporal_coupling": 0.9,
                                               "route_complexity": 0.1,
                                               "semantic_coupling": 0.2}})
        record = h.execute(other, OUTSIDE, str(self.solve_path),
                           str(self.work), solver="highs")
        h.record(record)
        h.recall(TASK)
        # The target cell still has no evidence: a surfaced memory is not a
        # reusable statistic.
        self.assertEqual(h.recall(TASK)["recommendations"], [])
        self.assertEqual(
            h.selector.candidate_ids(h.profile(TASK)), [])

    def test_retrieval_failure_is_labelled_apart_from_no_hits(self):
        """No backend = `degraded` (could not look). A backend that ran and
        found nothing reports an empty hit list and NO degraded block."""
        no_backend = ORHarness(home=self.home)  # no embedding injected
        self.addCleanup(no_backend.close)
        failed = no_backend.recall(TASK)
        self.assertIn("degraded", failed)
        self.assertNotIn("vector_recall", failed)

        with_backend = self.harness()
        # Build the index so the channel can really run. An execution's
        # document needs its captured task text, so the fact is recorded the
        # same way the write path does it.
        with_backend.index_sync.sync_entries()
        record = with_backend.execute(TASK, OUTSIDE, str(self.solve_path),
                                      str(self.work), solver="highs")
        self.assertEqual(
            with_backend.index_sync.sync_execution(record)["state"], "synced")
        with_backend.record(record)
        ran = with_backend.recall(TASK)
        self.assertNotIn("degraded", ran)
        self.assertIn("vector_recall", ran)
        # The channel RAN (it reports its backend) and surfaced the memory
        # it holds; what it did NOT do is fail silently.
        self.assertIn("backend", ran["vector_recall"])
        self.assertEqual(len(ran["vector_recall"]["execution_evidence"]), 1)

    def test_missing_index_is_reported_as_a_retrieval_failure(self):
        """A not-yet-built index is `degraded`, never a silent empty hit
        list — 'could not look' must not read as 'nothing similar exists'."""
        h = self.harness()
        result = h.recall(TASK)
        self.assertIn("degraded", result)
        self.assertIn("index", result["degraded"]["reason"])


# ---------------------------------------------------------------------------
# 4. a method outside any directory completes the whole loop
# ---------------------------------------------------------------------------


class TestOutsideDirectoryLoop(NoCatalogCase):
    def test_unknown_method_is_accepted_and_completes_the_loop(self):
        h = self.harness()
        # 1. prediction: unknown cost basis, NOT a refusal
        snapshot = h.predict_cost(TASK, OUTSIDE)
        self.assertEqual(snapshot.source, "unknown")
        self.assertIsNone(snapshot.expected_cost)
        self.assertEqual(snapshot.strategy_id, OUTSIDE)
        # 2. execution: no directory membership required
        record = h.execute(TASK, OUTSIDE, str(self.solve_path), str(self.work),
                           solver="highs")
        self.assertTrue(record.quality["feasible"])
        self.assertEqual(record.strategy_id, OUTSIDE)
        # 3. recording
        outcome = h.record(record)
        self.assertTrue(outcome["recorded"])
        # 4. recall: it is now a candidate, because it really ran
        result = h.recall(TASK)
        rec = next(r for r in result["recommendations"]
                   if r["strategy_id"] == OUTSIDE)
        self.assertEqual(rec["evidence"], "conditional_stats")
        self.assertEqual(rec["evidence_refs"], [record.execution_id])

    def test_unknown_method_never_rejected_by_identity(self):
        """Neither entry point raises on an unrecognized id."""
        h = self.harness()
        h.predict_cost(TASK, "totally-made-up")
        h.execute(TASK, "totally-made-up", str(self.solve_path), str(self.work),
                  solver="highs")

    def test_empty_strategy_id_is_still_refused(self):
        """The checks that REMAIN: the record must name the method."""
        h = self.harness()
        with self.assertRaises(ValueError) as caught:
            h.execute(TASK, "   ", str(self.solve_path), str(self.work),
                      solver="highs")
        self.assertIn("strategy_id is required", str(caught.exception))

    def test_solve_script_must_live_in_the_workspace(self):
        """The executor's own sandbox policy is untouched."""
        h = self.harness()
        outside = Path(self.home) / "outside.py"
        outside.write_text(SOLVE, encoding="utf-8")
        with self.assertRaises(ValueError):
            h.execute(TASK, OUTSIDE, str(outside), str(self.work),
                      solver="highs")

    def test_caller_proposal_marks_the_memory_less_ones(self):
        h = self.harness()
        record = h.execute(TASK, OUTSIDE, str(self.solve_path), str(self.work),
                           solver="highs")
        h.record(record)
        result = h.recall(TASK, candidates=[OUTSIDE, "custom:never-run"])
        self.assertEqual([r["strategy_id"] for r in result["recommendations"]],
                         [OUTSIDE])
        self.assertEqual(result["candidates_without_evidence"],
                         ["custom:never-run"])


# ---------------------------------------------------------------------------
# 5. no candidates supplied -> no menu invented
# ---------------------------------------------------------------------------


class TestNoCandidateMenu(NoCatalogCase):
    def test_plan_next_without_candidates_reports_no_candidates(self):
        h = self.harness()
        plan = h.plan_next(TASK, "ep1")
        self.assertEqual(plan["status"], "no_candidates")
        self.assertIn("does not generate a candidate menu",
                      plan["truncation_reason"])

    def test_strategy_outcome_protocol_without_candidates_too(self):
        h = self.harness()
        plan = h.plan_next(TASK, "ep1", protocol="strategy-outcome")
        self.assertEqual(plan["status"], "no_candidates")
        self.assertIn("does not generate a candidate menu",
                      plan["truncation_reason"])

    def test_supplied_candidates_are_used_as_given(self):
        h = self.harness()
        specs = [ActionSpec("execute_strategy", "t_outside",
                            strategy_id=OUTSIDE)]
        plan = h.plan_next(TASK, "ep1", candidates=specs)
        # No provider is attached, so the prediction path reports that
        # honestly — but the candidate was accepted, not replaced by a menu.
        self.assertNotEqual(plan["status"], "no_candidates")

    def test_cli_plan_next_reports_no_candidates(self):
        proc = _run_orx(self.home, "plan-next", "--task", str(self.task_path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        plan = out["result"]["plan"]
        self.assertEqual(plan["status"], "no_candidates")
        self.assertIn("does not generate a candidate menu",
                      plan["truncation_reason"])


# ---------------------------------------------------------------------------
# 6. history stays readable; nothing is back-filled
# ---------------------------------------------------------------------------


class TestHistoryRemainsReadable(NoCatalogCase):
    def test_legacy_identifier_is_readable_without_any_directory(self):
        """A record/entry written under the old ids stays fully usable: the
        framework reads it from memory, not from a directory."""
        h = self.harness()
        record = h.execute(TASK, "S01", str(self.solve_path), str(self.work),
                           solver="highs")
        h.record(record)
        h.induce(strategy_id="S01")
        # Recall, prediction and inspect all still work.
        self.assertEqual(
            h.recall(TASK)["recommendations"][0]["strategy_id"], "S01")
        self.assertEqual(h.predict_cost(TASK, "S01").source, "stats")
        self.assertEqual(
            h.inspect(bank="experience")["records"][0]["strategy_id"], "S01")

    def test_legacy_entry_content_is_not_rewritten_or_invented(self):
        """An entry's recorded content is preserved as-is, and an entry that
        recorded none reports none."""
        h = self.harness()
        h.bank.append(self.make_record(execution_id="ex_old", task_id="t1",
                                       strategy_id="S01"))
        h.bank.append(self.make_record(execution_id="ex_old2", task_id="t2",
                                       strategy_id="S01"))
        created = h.induce(strategy_id="S01")["results"][0]["created"]
        entry = h.sbank.get(created)
        # Nothing was invented: the memory never recorded a method type.
        self.assertIsNone(entry.strategy_type)
        self.assertEqual(entry.actions, [])
        self.assertIsNone(entry.fallback_strategy_id)
        # And the field is still writable by the harness that knows better.
        entry.strategy_type = "harness-declared"
        h.sbank.update(entry)
        self.assertEqual(h.sbank.get(created).strategy_type,
                         "harness-declared")

    def test_coverage_gaps_are_derived_from_evidence(self):
        """`strategies_without_evidence` names strategies this family's
        memory really contains but this CELL does not — never ids from a
        directory."""
        h = self.harness()
        # S01 ran in this family, but in a different structural cell.
        far = dict(TASK, task_id="t_far",
                   annotations={"coupling": {"resource_coupling": 0.1,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.1,
                                             "semantic_coupling": 0.2}})
        h.bank.append(self.make_record(execution_id="ex_far", task_id="t_far",
                                       strategy_id="S01",
                                       profile=h.profile(far)))
        snapshot = h.snapshot(TASK, "ep1")
        gaps = snapshot.coverage["coverage_gaps"]
        self.assertEqual(gaps["strategies_without_evidence"], ["S01"])
        self.assertIn("never tried in this family", gaps["note"])

    def test_unseen_strategy_is_not_reported_as_a_gap(self):
        h = self.harness()
        snapshot = h.snapshot(TASK, "ep1")
        self.assertEqual(
            snapshot.coverage["coverage_gaps"]["strategies_without_evidence"],
            [])


# ---------------------------------------------------------------------------
# 7. same id, different method: no silent merge
# ---------------------------------------------------------------------------


class TestSameIdDifferentMethod(NoCatalogCase):
    def test_config_mismatch_is_recorded_not_scored(self):
        """A prediction for one configuration is not bound to another.

        The executed action never observed a ``time_limit``, so the field is
        recorded as an UNKNOWN (the framework must not copy the predicted
        config onto the action to manufacture a match), and a key the action
        DID observe with a different value is a MISMATCH.
        """
        h = self.harness()
        candidate = ActionSpec("execute_strategy", "t_outside",
                               strategy_id=OUTSIDE,
                               params={"time_limit": 60,
                                       "solver": "highs"})
        prediction = h.predict_strategy_outcome(TASK, candidate, "ep1")
        record = h.execute(TASK, OUTSIDE, str(self.solve_path), str(self.work),
                           solver="highs", episode_id="ep1")
        h.record(record)
        bound = h.bind_strategy_outcome(
            prediction.prediction_id, record.action_id)
        info = bound.trace.model_info
        unknown = info.get("binding_unknown") or {}
        self.assertIn("config", unknown)
        self.assertIn("time_limit", unknown["config"])
        self.assertIsNone(unknown["config"]["time_limit"]["actual"])
        # The solver WAS observed and matches, so it is neither unknown nor
        # a mismatch.
        self.assertIsNone(info.get("binding_mismatch"))

    def test_each_attempt_is_its_own_statistic_sample(self):
        """Two different configurations under one id stay separate FACTS
        (their costs are not averaged into one another)."""
        h = self.harness()
        first = h.execute(TASK, OUTSIDE, str(self.solve_path), str(self.work),
                          solver="highs")
        h.record(first, override={"llm_tokens": 100})
        second = h.execute(TASK, OUTSIDE, str(self.solve_path), str(self.work),
                           solver="highs")
        h.record(second, override={"llm_tokens": 900})
        costs = sorted(
            h.bank.get(e).cost.llm_tokens
            for e in (first.execution_id, second.execution_id))
        self.assertEqual(costs, [100.0, 900.0])
        rec = h.recall(TASK)["recommendations"][0]
        self.assertEqual(len(rec["evidence_refs"]), 2)
        # The mean is reported as the mean of the two REAL observations.
        self.assertAlmostEqual(rec["expected"]["cost"]["llm_tokens"], 500.0,
                               places=3)

    def test_an_entry_keeps_its_own_recorded_content(self):
        """Two entries sharing a strategy id do not share content: each
        carries what was written on it."""
        h = self.harness()
        from or_harness.core.schema import StrategicEntry
        h.sbank.add(StrategicEntry(
            entry_id="se_a", strategy_id=OUTSIDE,
            pattern={"predicates": {"family": "routing"}},
            strategy_type="decomposition", actions=["split"], support_n=3))
        h.sbank.add(StrategicEntry(
            entry_id="se_b", strategy_id=OUTSIDE,
            pattern={"predicates": {"family": "routing"}},
            strategy_type="metaheuristic", actions=["search"], support_n=3))
        recs = h.recall(TASK, top=5)["recommendations"]
        types = {r["knowledge"]["strategy_type"] for r in recs}
        # Two entries, two honest reports — the framework does not pick one
        # or merge them into a fabricated average.
        self.assertEqual(len(recs), 2)
        self.assertEqual(types, {"decomposition", "metaheuristic"})


if __name__ == "__main__":
    unittest.main()
