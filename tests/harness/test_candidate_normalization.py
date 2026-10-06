"""Candidate entry-point normalization (r10).

The minimal candidate format (a top-level ``name``/``steps``, a method under
its own key, an allocated ``strategy_id``) must reach BOTH the direct
prediction entry and ``plan-next`` the same way, and the identity a missing
``strategy_id`` is allocated from must reflect the METHOD alone — never the
solver or the config. The behaviours pinned here:

1. top-level ``name``/``steps`` become ``method``;
2. a bare-string ``steps`` is ONE step;
3. a missing ``strategy_id`` is allocated DETERMINISTICALLY from the method;
4. the solver and the config do NOT change that identity (same method under a
   different solver keeps one id);
5. two disagreeing spellings of the name/steps are REFUSED before any call;
6. a generated ``strategy_id`` flows through plan-next into the candidate list.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.world_model.contracts import (  # noqa: E402
    CandidateRef,
    normalize_candidate_payload,
)


class TestCandidateNormalization(HarnessTestCase):

    def test_top_level_name_and_steps_become_method(self):
        out = normalize_candidate_payload({
            "name": "enumeration_with_constraints",
            "steps": ["derive a tight bound", "enumerate the reduced domain"],
            "solver": "python",
            "config": {"time_limit": 60}})
        ref = CandidateRef.from_dict(out)
        self.assertEqual(ref.method.get("name"), "enumeration_with_constraints")
        self.assertEqual(len(ref.method.get("steps") or []), 2)
        self.assertEqual(ref.solver, "python")
        self.assertEqual(ref.config.get("time_limit"), 60)

    def test_bare_string_steps_is_one_step(self):
        out = normalize_candidate_payload({"name": "m", "steps": "solve it"})
        self.assertEqual(out["method"]["steps"], ["solve it"])

    def test_missing_strategy_id_is_allocated_from_the_method(self):
        out = normalize_candidate_payload({"name": "m", "steps": ["s"]})
        self.assertTrue(out["strategy_id"].startswith("cand_"))
        self.assertTrue(out.get("normalization_notes"))

    def test_solver_and_config_do_not_change_the_identity(self):
        """The SAME method under a different solver / time_limit keeps ONE id.

        This is the whole point of hashing the method alone: hashing the
        tooling would scatter one method's history across several ids."""
        a = normalize_candidate_payload(
            {"name": "m", "steps": ["s1", "s2"], "solver": "highs",
             "config": {"time_limit": 10}})
        b = normalize_candidate_payload(
            {"name": "m", "steps": ["s1", "s2"], "solver": "cbc",
             "config": {"time_limit": 99}})
        self.assertEqual(a["strategy_id"], b["strategy_id"])

    def test_different_method_content_gets_a_different_id(self):
        a = normalize_candidate_payload({"name": "m1", "steps": ["s"]})
        b = normalize_candidate_payload({"name": "m2", "steps": ["s"]})
        self.assertNotEqual(a["strategy_id"], b["strategy_id"])

    def test_disagreeing_name_spellings_are_refused(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_candidate_payload({
                "name": "A", "method": {"name": "B", "steps": ["s"]}})
        self.assertIn("disagree", str(ctx.exception))

    def test_disagreeing_step_spellings_are_refused(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_candidate_payload({
                "steps": ["x"], "method": {"name": "A", "steps": ["y"]}})
        self.assertIn("steps", str(ctx.exception))

    def test_no_id_and_no_method_is_refused(self):
        with self.assertRaises(ValueError):
            normalize_candidate_payload({"action_type": "execute_strategy",
                                         "solver": "pulp"})

    def test_method_under_its_own_key_is_unified(self):
        out = normalize_candidate_payload(
            {"action_type": "execute_strategy",
             "method_planned": {"name": "planned", "steps": ["a"]}})
        self.assertEqual(out["method"]["name"], "planned")

    def test_solver_inside_method_is_lifted_to_the_candidate(self):
        out = normalize_candidate_payload(
            {"name": "m", "method": {"name": "m", "solver": "scip"}})
        self.assertEqual(out["solver"], "scip")
        self.assertNotIn("solver", out["method"])

    def test_legacy_actionspec_payload_is_passed_through(self):
        """A legacy ``ActionSpec`` (``measurement_scope``) keeps its own
        mapping — the normalizer must not re-interpret it."""
        raw = {"action_type": "execute_strategy", "strategy_id": "S01",
               "measurement_scope": "attempt", "params": {"time_limit": 5}}
        out = normalize_candidate_payload(raw)
        self.assertEqual(out, raw)


class TestPlanNextNormalization(HarnessTestCase):
    """The minimal candidate reaches ``plan-next`` and is normalized there."""

    def _task(self):
        return {"task_id": "t1", "family": "planning",
                "description": "min 50x+30y s.t. x+y<=1000, x>=2y+300",
                "spec": {"n_vars": 2, "n_constraints": 3, "n_int_vars": 2}}

    def test_minimal_candidate_is_accepted_and_identified(self):
        from or_harness.api import ORHarness
        from or_harness.world_model.provider import WorldModelProvider

        class StubProvider(WorldModelProvider):
            def describe(self):
                return {"provider": "stub"}

            def predict(self, request, timeout_s=None):
                return {"payload": {
                    "benefit": {"kind": "solution_quality",
                                "metric": "normalized_objective_gap",
                                "unit": "1-gap", "value": 0.9,
                                "baseline": {"kind": "conditional_stats",
                                             "value": 0.7}}},
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                        "error": None, "latency_s": 0.01}

        h = ORHarness(home=self.home, world_model=StubProvider())
        self.addCleanup(h.close)
        plan = h.plan_next(self._task(), episode_id="ep1", candidates=[
            {"name": "enum", "steps": ["bound", "enumerate"],
             "solver": "python"},
        ])
        specs = plan.get("candidates") or []
        self.assertEqual(len(specs), 1)
        sid = specs[0]["action_spec"]["strategy_id"]
        self.assertTrue(str(sid).startswith("cand_"))
        self.assertEqual(
            specs[0]["action_spec"]["method"]["name"], "enum")

    def test_ill_formed_candidate_is_reported_not_a_doomed_call(self):
        from or_harness.api import ORHarness
        from or_harness.world_model.provider import WorldModelProvider

        class StubProvider(WorldModelProvider):
            def describe(self):
                return {"provider": "stub"}

            def predict(self, request, timeout_s=None):
                return {"payload": {}, "usage": {}, "error": None,
                        "latency_s": 0.01}

        h = ORHarness(home=self.home, world_model=StubProvider())
        self.addCleanup(h.close)
        plan = h.plan_next(self._task(), episode_id="ep1", candidates=[
            {"name": "A", "method": {"name": "B", "steps": ["s"]}},
        ])
        rejections = plan.get("candidate_rejections") or []
        self.assertTrue(rejections)
        self.assertIn("disagree", rejections[0])


if __name__ == "__main__":
    unittest.main()
