"""Experiment runner: task streams, four ablation modes, metrics CSV.

The runner simulates a harness executing a stream of tasks T1..Tn and answers
the research question: does the agent learn which strategies fit which
structures, achieving similar or better quality at lower cost over time?

The C vs D ablation (strategic vs cost-aware) only separates in "quality tied,
cost divergent" scenarios — the experimental support for the thesis that cost
awareness is a necessary component of memory, not an optional extra.

Synthetic tasks are generated with parameterized family/scale/coupling, and
each (strategy, family-structure) pair has a hidden ground-truth
quality/cost law that the executions sample deterministically (seeded), so
the learning curves are reproducible. Executions go through the real sandbox
(generated solve scripts) — the runner reuses the actual system end to end.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from or_harness.api import ORHarness
from or_harness.core.schema import COST_DIMENSIONS


# ---------------------------------------------------------------------------
# synthetic task stream
# ---------------------------------------------------------------------------

FAMILIES = ("routing", "scheduling", "assignment")

#: Hidden ground truth per (family_cluster, strategy): (quality, cost_scale).
#: Structured so that the best strategy differs by family, quality is often
#: tied while cost diverges (the C/D separation scenario), and some priors
#: are wrong (memory must correct them).
GROUND_TRUTH: Dict[str, Dict[str, Dict[str, float]]] = {
    "routing": {
        "S01": {"quality": 0.70, "cost": 3.0},
        "S04": {"quality": 0.70, "cost": 1.0},   # tied quality, 3x cheaper
        "S06": {"quality": 0.55, "cost": 0.8},
        "S08": {"quality": 0.90, "cost": 6.0},   # best quality, pricey
    },
    "scheduling": {
        "S01": {"quality": 0.62, "cost": 3.0},
        "S03": {"quality": 0.85, "cost": 6.0},   # tied quality with S05, pricey
        "S04": {"quality": 0.40, "cost": 1.0},
        "S05": {"quality": 0.85, "cost": 2.0},
    },
    "assignment": {
        "S01": {"quality": 0.78, "cost": 2.5},
        "S06": {"quality": 0.78, "cost": 0.6},   # tied quality, much cheaper
        "S07": {"quality": 0.82, "cost": 3.5},
        "S04": {"quality": 0.58, "cost": 1.0},
    },
}

COUPLING_BY_FAMILY = {
    "routing": {"resource_coupling": 0.9, "temporal_coupling": 0.1,
                "route_complexity": 0.85, "semantic_coupling": 0.8},
    "scheduling": {"resource_coupling": 0.3, "temporal_coupling": 0.9,
                   "route_complexity": 0.2, "semantic_coupling": 0.6},
    "assignment": {"resource_coupling": 0.8, "temporal_coupling": 0.3,
                   "route_complexity": 0.4, "semantic_coupling": 0.7},
}

#: The SIMULATED HARNESS's own candidate set, per family — an explicit
#: ablation fixture, not a framework directory. In a real run the outer agent
#: names the methods it wants compared; here the runner plays that role and
#: proposes exactly the strategies its hidden ground truth covers. The
#: framework never sees this list except as the caller's proposal, and it
#: never adds to it: a strategy the harness does not propose is not scored.
CANDIDATES_BY_FAMILY: Dict[str, List[str]] = {
    family: sorted(laws) for family, laws in GROUND_TRUTH.items()
}

#: The method the simulated harness falls back to when it consults NO memory
#: (``memory_mode="none"``): its own declared default, chosen by the caller.
#: This is the ``none`` arm of the ablation — it exists to show what a harness
#: that ignores its memory does, and it is a fixture decision, not a
#: framework-supplied recommendation.
DEFAULT_STRATEGY_BY_FAMILY: Dict[str, str] = {
    family: candidates[0]
    for family, candidates in CANDIDATES_BY_FAMILY.items()
}


@dataclass
class SyntheticTask:
    task_id: str
    family: str
    spec: Dict[str, Any]

    def to_task_json(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "family": self.family,
                "spec": self.spec,
                "annotations": {"coupling": COUPLING_BY_FAMILY[self.family]}}


def make_task_stream(n_tasks: int = 30, seed: int = 7) -> List[SyntheticTask]:
    """Deterministic stream cycling families with mild scale variation."""
    tasks = []
    for i in range(n_tasks):
        family = FAMILIES[i % len(FAMILIES)]
        jitter = int(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:4], 16)
        tasks.append(SyntheticTask(
            task_id=f"T{i + 1:03d}", family=family,
            spec={"n_vars": 500 + (jitter % 1500),
                  "n_constraints": 200 + (jitter % 800),
                  "n_int_vars": 300 + (jitter % 900),
                  "density": 0.01}))
    return tasks


def _execution_law(task: SyntheticTask, strategy_id: str) -> Dict[str, float]:
    """Deterministic 'true' outcome of running strategy on task."""
    truth = GROUND_TRUTH.get(task.family, {}).get(
        strategy_id, {"quality": 0.45, "cost": 5.0})  # unknown pairs: bad+slow
    jitter = (int(hashlib.sha256(
        f"{task.task_id}:{strategy_id}".encode()).hexdigest()[:6], 16) % 100) / 100.0
    quality = max(0.05, min(0.99, truth["quality"] + (jitter - 0.5) * 0.04))
    cost_scale = truth["cost"] * (1.0 + (jitter - 0.5) * 0.1)
    return {"quality": quality, "cost_scale": cost_scale}


def _solve_script(quality: float, cost_scale: float) -> str:
    """A real solve script whose result.json encodes the ground-truth outcome.

    Solver runtime follows the cost law directly (deterministic) so cost
    comparisons do not drown in sub-process wall-clock noise."""
    gap = round(1.0 - quality, 6)
    runtime = round(2.0 * cost_scale, 6)
    objective = round(1000.0 * (1.0 - gap), 4)
    bound = 1000.0
    return (
        "import json\n"
        f"with open('result.json', 'w') as fh:\n"
        f"    json.dump({{'status': 'feasible', 'objective_value': {objective},\n"
        f"               'objective_bound': {bound}, 'runtime_seconds': {runtime}}}, fh)\n"
    )


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

MEMORY_MODES = ("none", "cases", "strategic", "cost-aware")

#: M6 prediction modes — ORTHOGONAL to the memory ablation above. The two
#: axes answer different questions: MEMORY_MODES asks "which memory layer
#: is consulted", PREDICTION_MODES asks "what does the world model predict,
#: and may its knowledge prediction influence the choice".
PREDICTION_MODES = ("x-b-only", "h-x-b", "h-x-b-value")


@dataclass
class RunMetrics:
    mode: str
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def write_csv(self, path: Path) -> None:
        if not self.rows:
            return
        fieldnames = ["task_index", "task_id", "family", "strategy_id",
                      "quality", "feasible"] + [f"cost_{d}" for d in COST_DIMENSIONS] + \
                     ["cumulative_cost_scalar", "memory_entries", "memory_executions",
                      "induction_hints"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)


def run_stream(mode: str, tasks: Sequence[SyntheticTask], home: str, *,
               cost_weights: Optional[Dict[str, float]] = None,
               induce_every: int = 5,
               warmup: bool = True) -> RunMetrics:
    """Run one ablation mode over the task stream.

    The simulated harness plays the OUTER AGENT's role: it proposes the
    methods it is considering (``CANDIDATES_BY_FAMILY`` — its own declared
    hypothesis set), asks the framework to recall what memory holds for them,
    executes the best-scoring one, records with llm_tokens backfilled from the
    cost law, and induces every ``induce_every`` tasks (an explicit harness
    decision).

    Under ``memory_mode="none"`` the harness consults no memory at all, so
    recall returns nothing and it runs its OWN declared default
    (``DEFAULT_STRATEGY_BY_FAMILY``). That is the point of the ``none`` arm:
    it shows what an agent that ignores its memory does. It is not a
    framework recommendation — the framework has no default to offer.

    ``warmup`` executes every (family, strategy) pair once BEFORE the measured
    stream, mirroring a harness's exploration phase. This aligns the evidence
    base across ablation modes so the measured differences come from the
    selection rule alone, not from divergent exploration trajectories (a
    greedy top-1 agent never revisits strategies the cold start ranked low).
    """
    if mode not in MEMORY_MODES:
        raise ValueError(f"mode must be one of {MEMORY_MODES}")
    harness = ORHarness(home=home, cost_weights=cost_weights)
    metrics = RunMetrics(mode=mode)
    cumulative = 0.0
    workdir = Path(tempfile.mkdtemp(prefix="orx_exp_"))
    try:
        if warmup:
            _warmup(harness, workdir)
        for index, task in enumerate(tasks):
            task_json = task.to_task_json()
            proposed = CANDIDATES_BY_FAMILY[task.family]
            recs = harness.recall(task_json, top=1, memory_mode=mode,
                                  candidates=proposed)
            if recs["recommendations"]:
                strategy_id = recs["recommendations"][0]["strategy_id"]
            else:
                # No memory consulted (or nothing recalled for the proposed
                # methods): the harness falls back to ITS OWN default. The
                # framework supplies no menu here.
                strategy_id = DEFAULT_STRATEGY_BY_FAMILY[task.family]
            law = _execution_law(task, strategy_id)
            script = workdir / f"solve_{task.task_id}.py"
            script.write_text(_solve_script(law["quality"], law["cost_scale"]),
                              encoding="utf-8")
            # Freeze the pre-execution cost prediction actually used, so
            # record-time feedback compares against it (never a post-hoc
            # estimate).
            prediction = harness.predict_cost(task_json, strategy_id)
            record = harness.execute(task_json, strategy_id, str(script),
                                     str(workdir), solver="highs")
            llm_tokens = 1500.0 * law["cost_scale"]
            # tool_calls counts ALL tool invocations in the attempt's scope
            # (writing the script, running it, reading the result) — the
            # simulated harness knows this, the sandbox cannot see it.
            outcome = harness.record(
                record,
                override={"llm_tokens": llm_tokens, "tool_calls": 3.0},
                prediction=prediction)
            record = harness.bank.get(record.execution_id)  # post-backfill fact
            scalar = record.cost.scalarize(
                cost_weights or harness.selector.cost_weights)
            cumulative += scalar
            metrics.rows.append({
                "task_index": index,
                "task_id": task.task_id,
                "family": task.family,
                "strategy_id": strategy_id,
                "quality": round(max(0.0, 1.0 - (record.quality.get("gap") or 1.0)), 4),
                "feasible": int(bool(record.quality.get("feasible"))),
                **{f"cost_{d}": round(getattr(record.cost, d), 4)
                   for d in COST_DIMENSIONS},
                "cumulative_cost_scalar": round(cumulative, 4),
                "memory_entries": harness.sbank.count(),
                "memory_executions": harness.bank.count(),
                "induction_hints": len(outcome["induction_hints"]),
            })
            if (index + 1) % induce_every == 0:
                harness.induce(all_=True)
    finally:
        harness.close()
    return metrics


def _warmup(harness: ORHarness, workdir: Path) -> None:
    """Deterministic exploration: run every (family, strategy) pair once and
    record it. All ablation modes share this evidence base."""
    for family in FAMILIES:
        task = SyntheticTask(task_id=f"warm_{family}", family=family,
                             spec={"n_vars": 800, "n_constraints": 400,
                                   "n_int_vars": 500, "density": 0.01})
        task_json = task.to_task_json()
        for strategy_id in GROUND_TRUTH[family]:
            law = _execution_law(task, strategy_id)
            script = workdir / f"warm_{family}_{strategy_id}.py"
            script.write_text(_solve_script(law["quality"], law["cost_scale"]),
                              encoding="utf-8")
            record = harness.execute(task_json, strategy_id, str(script),
                                     str(workdir), solver="highs")
            harness.record(record,
                           override={"llm_tokens": 1500.0 * law["cost_scale"],
                                     "tool_calls": 3.0})
    harness.induce(all_=True)


def run_ablation(output_dir: str, n_tasks: int = 30, seed: int = 7,
                 modes: Sequence[str] = MEMORY_MODES) -> Dict[str, Any]:
    """Run all four ablation modes over identical task streams and write one
    metrics CSV per mode plus a combined summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = make_task_stream(n_tasks=n_tasks, seed=seed)
    summary: Dict[str, Any] = {"n_tasks": n_tasks, "seed": seed, "modes": {}}
    for mode in modes:
        metrics = run_stream(mode, tasks, home=str(out / f"home_{mode}"))
        csv_path = out / f"metrics_{mode}.csv"
        metrics.write_csv(csv_path)
        if metrics.rows:
            n = len(metrics.rows)
            half = max(1, n // 2)
            late = metrics.rows[half:]
            early = metrics.rows[:half]
            summary["modes"][mode] = {
                "csv": str(csv_path),
                "mean_quality_early": round(sum(r["quality"] for r in early) / len(early), 4),
                "mean_quality_late": round(sum(r["quality"] for r in late) / len(late), 4),
                "final_cumulative_cost": metrics.rows[-1]["cumulative_cost_scalar"],
                "mean_cost_per_task": round(
                    metrics.rows[-1]["cumulative_cost_scalar"] / n, 4),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    return summary
