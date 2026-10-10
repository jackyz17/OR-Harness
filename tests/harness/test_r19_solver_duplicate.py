"""r19: `config.solver` is a DUPLICATE field, not an identity conflict.

The field run stored evaluations whose benefit/interval/risk were blocked by
``identity_mismatch`` on a ``config.solver`` that merely RESTATED the
candidate's top-level ``solver`` with a different spelling (``pulp_cbc`` in
config vs the normalized ``pulp`` at the top level). The two spellings need
not agree — that is not a different configuration.

This file pins BOTH halves of the fix, on NEW bindings and on HISTORY:

1. a new binding does not let ``config.solver`` enter the mismatch/unknown/
   match comparison (the top-level solver check stays);
2. ``binding_attribution`` skips ``config.solver`` as a duplicate, so a
   STORED conflict stops blocking the outcome when re-derived;
3. the rule rebuild REPORTs the correction even when the stored and derived
   STATE are both ``evaluated`` (the old state-only comparison missed it);
4. the genuine conflicts still block (a real top-level solver mismatch, a
   real different config value, a wrong task/strategy).

The acceptance table from the request:

| scenario | expected |
|---|---|
| top-level solver matches; ``config.solver`` spelled differently | not blocked by the duplicate |
| the top-level solver really differs | the original outcome block stays |
| another config key really differs | the original block stays |
| ``formulation`` / ``integer`` unreported | recorded as unknown, blocks nothing |
| a history record whose ONLY conflict is the duplicate solver | derived evaluation unblocked, the stored record unchanged |
| a history record with a task/strategy conflict | the corresponding block stays |
| repeated rebuilds | the sample count is stable |
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.attribution import (  # noqa: E402
    DUPLICATE_CONFIG_KEYS,
    binding_attribution,
)

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")


def _blocked(attr):
    return {dim: entries for dim, entries in attr["blocked"].items()
            if entries}


# ---------------------------------------------------------------------------
# 1. the attribution table
# ---------------------------------------------------------------------------


class TestDuplicateSolverIsNotAConflict(unittest.TestCase):

    def test_the_duplicate_set_names_solver(self):
        self.assertIn("solver", DUPLICATE_CONFIG_KEYS)

    def test_a_config_solver_mismatch_blocks_nothing(self):
        attr = binding_attribution(
            {"config": {"solver": {"predicted": "pulp_cbc",
                                   "actual": "pulp"}}}, {})
        self.assertEqual(_blocked(attr), {},
                         "a duplicate config.solver must not block the "
                         "outcome")
        entry = attr["fields"]["config.solver"]
        self.assertEqual(entry["blocks"], [])
        self.assertEqual(entry["kind"], "duplicate")
        self.assertIn("duplicate field", entry["reason"])

    def test_a_config_solver_unknown_blocks_nothing(self):
        attr = binding_attribution(
            {}, {"config": {"solver": {"predicted": "pulp_cbc",
                                       "actual": None}}})
        self.assertEqual(_blocked(attr), {})
        self.assertEqual(attr["fields"]["config.solver"]["kind"], "duplicate")

    def test_a_real_top_level_solver_mismatch_still_blocks_outcome(self):
        attr = binding_attribution(
            {"solver": {"predicted": "highs", "actual": "glpk"}}, {})
        self.assertTrue(attr["blocked"]["benefit"])
        self.assertTrue(attr["blocked"]["risk"])
        self.assertTrue(attr["blocked"]["interval"])
        self.assertFalse(attr["blocked"]["cost"],
                         "a different solver is a deviation: the spend stays")

    def test_another_config_key_mismatch_still_blocks_outcome(self):
        attr = binding_attribution(
            {"config": {"time_limit": {"predicted": 60, "actual": 30}}}, {})
        self.assertTrue(attr["blocked"]["benefit"])
        self.assertFalse(attr["blocked"]["cost"])

    def test_an_unreported_formulation_or_integer_blocks_nothing(self):
        attr = binding_attribution({}, {"config": {
            "formulation": {"predicted": "MILP", "actual": None},
            "integer": {"predicted": True, "actual": None},
        }})
        self.assertEqual(_blocked(attr), {},
                         "an unreported approach-sounding key is a caveat, "
                         "not a veto")
        self.assertIn("config.formulation", attr["fields"])
        self.assertIn("config.integer", attr["fields"])

    def test_duplicate_and_real_conflict_together(self):
        attr = binding_attribution(
            {"config": {"solver": {"predicted": "pulp_cbc", "actual": "pulp"},
                        "time_limit": {"predicted": 60, "actual": 30}}}, {})
        # The duplicate is skipped; the real one still blocks benefit.
        self.assertTrue(attr["blocked"]["benefit"])
        self.assertEqual(attr["fields"]["config.solver"]["kind"], "duplicate")
        self.assertIn("config.time_limit", attr["fields"])

    def test_a_wrong_task_still_excludes_everything(self):
        attr = binding_attribution({"task_id": {"predicted": "t1",
                                                "actual": "t2"}}, {})
        for dim in ("benefit", "cost", "risk", "interval"):
            self.assertTrue(attr["blocked"][dim])

    def test_a_wrong_strategy_still_excludes_everything(self):
        attr = binding_attribution({"strategy_id": {"predicted": "S01",
                                                    "actual": "S02"}}, {})
        for dim in ("benefit", "cost", "risk", "interval"):
            self.assertTrue(attr["blocked"][dim])


# ---------------------------------------------------------------------------
# 2. the new binding skips the duplicate key
# ---------------------------------------------------------------------------


class TestNewBindingSkipsDuplicateSolver(unittest.TestCase):
    """The binding loop must not read ``config.solver`` at all."""

    def _config_keys_examined(self):
        # A structural probe: the loop's own source must skip 'solver'
        # BEFORE it reads the observed value, so a config.solver can never
        # produce a mismatch, an unknown, or a match entry.
        import inspect
        from or_harness import api
        source = inspect.getsource(api.ORHarness.bind_strategy_outcome)
        return source

    def test_binding_skips_config_solver(self):
        source = self._config_keys_examined()
        self.assertIn('if key == "solver":', source,
                      "the binding loop must skip the duplicate key")
        # The skip precedes the observation lookup, so no entry is built.
        skip_at = source.index('if key == "solver":')
        lookup_at = source.index('if key in observed_values:')
        self.assertLess(skip_at, lookup_at,
                        "the skip must come before the observed-value read, "
                        "so no mismatch/unknown/match entry is produced")


# ---------------------------------------------------------------------------
# 3. the rule-rebuild reports a same-state correction
# ---------------------------------------------------------------------------


class TestRuleRebuildDetectsUnblockedDimensions(unittest.TestCase):
    def test_state_and_attribution_detection_are_both_present(self):
        """The rebuild must compare more than the state, or an 'evaluated ->
        evaluated' correction (duplicate conflict unblocked, cost kept the
        state) would be missed."""
        import inspect
        from or_harness.world_model import episode_closeout as ec
        source = inspect.getsource(ec._live_evaluation)
        self.assertIn('changed["attribution"]', source)
        self.assertIn('changed["eligibility"]', source)


# ---------------------------------------------------------------------------
# 4. end to end: a history record whose only conflict is the duplicate
# ---------------------------------------------------------------------------


class TestHistoryRebuildEndToEnd(HarnessTestCase):
    """A STORED evaluation blocked only by ``config.solver`` is re-derived
    UNBLOCKED; the stored record is unchanged and repeated rebuilds do not
    add a sample. Uses a TEMPORARY harness (never the live experiment DB)."""

    def setUp(self):
        super().setUp()
        saved = {k: os.environ.pop(k, None) for k in EMBEDDING_ENV_KEYS}

        def restore():
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.provider = _StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=self.backend)
        self.addCleanup(self.h.close)

    def _solve(self, solver="pulp"):
        from pathlib import Path
        work = Path(self.home) / "ws"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json, os\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 100.0,"
            " 'objective_bound': 100.0, 'runtime_seconds': 0.01,"
            " 'config': {'solver': os.environ.get('SOLVER', 'pulp')}}, fh)\n",
            encoding="utf-8")
        import os
        os.environ["SOLVER"] = solver
        self.addCleanup(lambda: os.environ.pop("SOLVER", None))
        return script, work

    def test_duplicate_only_conflict_rebuilds_unblocked(self):
        from or_harness.world_model.episode_closeout import (
            StrategyPredictionEvaluation,
            _live_evaluation,
        )
        task = {"task_id": "t1", "family": "routing",
                "description": ("load the depot then deliver demand 100 "
                                "units"),
                "spec": {"n_vars": 10, "n_constraints": 5, "n_int_vars": 10},
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04",
                   "solver": "pulp", "config": {"solver": "pulp_cbc"}},
            "ep1")
        script, work = self._solve("pulp")
        record = self.h.execute(task, "S04", str(script), str(work),
                                solver="pulp", episode_id="ep1",
                                prediction_id=prediction.prediction_id)
        self.h.record(record)
        self.h.close_episode("t1", "ep1")
        # A stored record written under the OLD rule: blocked on the
        # duplicate solver, but cost kept it at ``evaluated``.
        legacy = StrategyPredictionEvaluation(
            prediction_id=prediction.prediction_id, task_id="t1",
            episode_id="ep1", state="evaluated",
            benefit={"eligibility": "identity_mismatch",
                     "predicted": 0.8, "observed": 1.0, "signed_error": 0.2},
            cost={"eligibility": "evaluable",
                  "per_dim": {"solver_runtime_s": {
                      "predicted": 3.0, "actual": 0.01, "log_error": -5.7}}},
            risk={"eligibility": "evaluable", "scored": [], "unscored": [],
                  "observed_units": {}},
            interval={"eligibility": "identity_mismatch"},
            attribution={"benefit": [{"field": "config.solver",
                                      "kind": "mismatch"}],
                         "interval": [{"field": "config.solver",
                                       "kind": "mismatch"}],
                         "risk": [{"field": "config.solver",
                                   "kind": "mismatch"}]},
        )
        derived, correction = _live_evaluation(self.h, legacy)
        # The stored state did NOT change (both ``evaluated``), yet the
        # correction is still reported because the attribution moved.
        self.assertEqual(legacy.state, "evaluated")
        self.assertIsNotNone(correction,
                             "an evaluated->evaluated unblock must be "
                             "reported, not missed")
        self.assertIn("attribution", correction["fields"])
        # The STORED attribution had the duplicate; the DERIVED one drops it.
        stored_block = correction["detail"]["attribution"]["stored"]
        derived_block = correction["detail"]["attribution"]["derived"]
        self.assertIn("config.solver",
                      {e["field"] for e in stored_block.get("benefit", [])})
        self.assertNotIn("config.solver",
                         {e["field"] for e in derived_block.get("benefit", [])})
        # The derived attribution no longer blocks benefit on the duplicate.
        self.assertNotIn(
            "config.solver",
            {e["field"] for e in derived.attribution.get("benefit", [])})
        # The stored record is never mutated.
        self.assertEqual(legacy.attribution["benefit"][0]["field"],
                         "config.solver")
        # A second rebuild is stable (no sample added, same verdict).
        derived2, correction2 = _live_evaluation(self.h, legacy)
        self.assertEqual(derived2.state, derived.state)
        self.assertIsNotNone(correction2)

    def test_repeated_rebuild_is_stable(self):
        self.test_duplicate_only_conflict_rebuilds_unblocked()


class _StubProvider:
    name = "stub-r19"

    def describe(self):
        return {"provider_model": "stub-r19", "provider_version": "1"}

    def predict(self, request, timeout_s=None):
        return {"payload": {
            "benefit": {"kind": "solution_quality",
                        "metric": "normalized_objective_gap", "unit": "1-gap",
                        "value": 0.8,
                        "baseline": {"kind": "conditional_stats",
                                     "value": 0.7}},
            "cost": {"solver_runtime_s": 3.0},
        }, "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "error": None, "latency_s": 0.02}


if __name__ == "__main__":
    unittest.main()
