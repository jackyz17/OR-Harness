"""Experiments runner tests: four ablation modes produce metrics, memory
improves decisions, and cost-aware mode separates from strategic mode in the
quality-tied/cost-divergent scenario."""
import csv
import unittest
from pathlib import Path

from helpers import HarnessTestCase

from or_harness.experiments.runner import (
    GROUND_TRUTH,
    make_task_stream,
    run_ablation,
    run_stream,
)


class TestRunner(HarnessTestCase):
    def test_task_stream_deterministic(self):
        a = make_task_stream(9, seed=3)
        b = make_task_stream(9, seed=3)
        self.assertEqual([t.to_task_json() for t in a],
                         [t.to_task_json() for t in b])
        families = {t.family for t in a}
        self.assertEqual(families, {"routing", "scheduling", "assignment"})

    def test_run_stream_produces_rows(self):
        tasks = make_task_stream(12, seed=7)
        metrics = run_stream("cost-aware", tasks, home=self.home)
        self.assertEqual(len(metrics.rows), 12)
        self.assertTrue(all(0.0 <= r["quality"] <= 1.0 for r in metrics.rows))
        # Cumulative cost grows monotonically.
        cum = [r["cumulative_cost_scalar"] for r in metrics.rows]
        self.assertEqual(cum, sorted(cum))

    def test_memory_learns_cheap_tied_strategy(self):
        # routing: S01 and S04 tie on quality, S04 is 3x cheaper. With memory,
        # later routing tasks should converge to S04; with none they cannot.
        tasks = make_task_stream(24, seed=11)
        aware = run_stream("cost-aware", tasks, home=str(Path(self.home) / "a"))
        none_ = run_stream("none", tasks, home=str(Path(self.home) / "n"))
        aware_late_routing = [r for r in aware.rows[12:] if r["family"] == "routing"]
        self.assertTrue(aware_late_routing)
        cheap_picks = sum(1 for r in aware_late_routing if r["strategy_id"] == "S04")
        self.assertGreaterEqual(cheap_picks, len(aware_late_routing) - 1)
        # mode none always plays the default strategy
        self.assertTrue(all(r["strategy_id"] == "S01" for r in none_.rows))
        # and pays more for it
        self.assertLess(aware.rows[-1]["cumulative_cost_scalar"],
                        none_.rows[-1]["cumulative_cost_scalar"])

    def test_strategic_vs_cost_aware_separation(self):
        # The C/D ablation: quality-tied/cost-divergent structure means
        # cost-aware should end with lower cumulative cost than strategic.
        tasks = make_task_stream(24, seed=5)
        strategic = run_stream("strategic", tasks, home=str(Path(self.home) / "s"))
        aware = run_stream("cost-aware", tasks, home=str(Path(self.home) / "d"))
        self.assertLessEqual(aware.rows[-1]["cumulative_cost_scalar"],
                             strategic.rows[-1]["cumulative_cost_scalar"])

    def test_ablation_writes_csvs_and_summary(self):
        out = Path(self.home) / "exp"
        summary = run_ablation(str(out), n_tasks=12, seed=7)
        self.assertEqual(set(summary["modes"].keys()),
                         {"none", "cases", "strategic", "cost-aware"})
        for mode in summary["modes"]:
            csv_path = Path(summary["modes"][mode]["csv"])
            self.assertTrue(csv_path.exists())
            with open(csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 12)
            self.assertIn("cost_llm_tokens", rows[0])
            self.assertIn("cumulative_cost_scalar", rows[0])
        self.assertTrue((out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
