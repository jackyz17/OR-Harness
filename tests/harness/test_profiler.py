"""Profiler tests: determinism, harness-supplied channel, derived channel."""
import unittest

from helpers import HarnessTestCase

from or_harness.profiling.profiler import profile_task


class TestProfiler(HarnessTestCase):
    def test_requires_id_and_family(self):
        with self.assertRaises(ValueError):
            profile_task({"family": "routing"})
        with self.assertRaises(ValueError):
            profile_task({"task_id": "t1"})

    def test_harness_supplied_channel(self):
        profile = profile_task({
            "task_id": "t1", "family": "routing",
            "annotations": {"coupling": {"resource_coupling": 0.9,
                                         "temporal_coupling": 0.2}},
        })
        self.assertEqual(profile.source, "harness_supplied")
        self.assertEqual(profile.resource_coupling, 0.9)
        self.assertEqual(profile.temporal_coupling, 0.2)
        self.assertIsNone(profile.route_complexity)

    def test_top_level_coupling_keys_accepted(self):
        profile = profile_task({"task_id": "t1", "family": "routing",
                                "resource_coupling": 0.7})
        self.assertEqual(profile.source, "harness_supplied")
        self.assertEqual(profile.resource_coupling, 0.7)

    def test_derived_from_spec_only(self):
        task = {
            "task_id": "t2", "family": "scheduling",
            "spec": {"n_vars": 500, "time_periods": 40,
                     "resources": ["crew", "machine"],
                     "entities": [{"name": "x", "indexes": ["i", "t"]},
                                  {"name": "y", "indexes": ["j"]}]},
        }
        p1 = profile_task(task)
        p2 = profile_task(task)
        self.assertEqual(p1.to_dict(), p2.to_dict())  # determinism
        self.assertEqual(p1.source, "derived")
        self.assertIsNotNone(p1.temporal_coupling)
        self.assertGreater(p1.temporal_coupling, 0.5)
        self.assertEqual(p1.scale_features["n_vars"], 500.0)

    def test_derived_from_code_ast(self):
        code = """
import json
x = model.addVars(nodes, time_periods)
cap = model.addVars(resources, time_periods)
for t in time_periods:
    model.addConstr(sum(x[i, t] for i in nodes) <= cap[0, t])
with open("result.json", "w") as fh:
    json.dump({"status": "optimal"}, fh)
"""
        task = {"task_id": "t3", "family": "assignment", "spec": {}}
        p = profile_task(task, code=code)
        self.assertEqual(p.source, "derived")
        self.assertIsNotNone(p.temporal_coupling)
        self.assertIsNotNone(p.resource_coupling)

    def test_same_input_same_output_with_code(self):
        task = {"task_id": "t4", "family": "assignment",
                "spec": {"time_periods": 10}}
        code = "x = y[1]\n"
        self.assertEqual(profile_task(task, code).to_dict(),
                         profile_task(task, code).to_dict())

    def test_clamps_out_of_range(self):
        profile = profile_task({"task_id": "t5", "family": "f",
                                "resource_coupling": 7.5})
        self.assertEqual(profile.resource_coupling, 1.0)


if __name__ == "__main__":
    unittest.main()
