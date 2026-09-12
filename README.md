# OR-Harness

A small, composable, harness-friendly **strategy-learning capability layer** for operations-research agents.

> Given an evolving stream of large-scale industrial optimization tasks, can an OR agent learn from previous executions which solving strategies are appropriate for particular problem structures, and increasingly achieve similar or better solution quality at lower execution cost?

OR-Harness runs *inside* an outer harness agent (Hermes-style). It is not an autonomous agent: no conversation loop, no runtime LLM calls, no hidden global state. The outer agent orchestrates; this layer advises, executes, and remembers.

## What it provides

- **Two-layer memory**: an append-only **Execution Evidence Bank** (episodic facts: actual strategy/quality/cost, failures, artifacts; lossy compaction deferred) and a derived **Strategic Knowledge Bank** (commitments: expected quality/cost/failure risk, with prediction intervals and forward validation; admission never depends on evidence survival — re-inducible from retained evidence). Statistics are computed on the fly, never persisted.
- **Deterministic profiling**: structural coupling features from your task spec or supplied annotations — no NLP subsystem.
- **Strategy selection with full control**: transparent scoring (`α·Q̂ − β·C_scalar − γ·R̂`), two-layer evidence fallback, four ablation modes.
- **Sandboxed execution**: AST policy + POSIX rlimits + wall-clock timeout for your solve scripts; basic verification; five-dimensional cost metering (retries count).
- **Induction you control**: C1–C6 evidence hints after every record; `induce` is always your explicit call. Scope ladder L1→L2→L3 with falsifiable widening; cold archive with anti-resurrection.
- **Seven solver adapters** (highs, pulp, ortools, scip, copt, pyomo, gurobi) — availability probing only; you pick the concrete solver per situation.

## Quick start

```bash
pip install -e .                # zero runtime dependencies (pure stdlib)
pip install -e ".[solvers-free]"  # optional: highspy + pulp

orx doctor                       # probe solvers, check memory home
orx profile   --task t.json
orx recall    --task t.json --top 3
orx execute   --task t.json --strategy S04 --code solve.py --workspace ws --solver highs
orx record    --execution exec.json --override llm_tokens=1840
orx induce    --strategy S04
orx inspect   --bank strategic
```

Memory lives in an explicit directory (`--home` or `$OR_HARNESS_HOME`), stored as a single SQLite file.

## Documentation

- **[SKILL.md](SKILL.md)** — the thin contract for harness agents (start here)
- **[references/concepts.md](references/concepts.md)** — two-layer memory, CostVector, disposal ladder
- **[references/induction.md](references/induction.md)** — C1–C6, scope ladder, forward validation
- **[references/examples.md](references/examples.md)** — three complete walkthroughs

## Experiments

```bash
PYTHONPATH=src python3 -m or_harness.experiments.runner  # or via the API
```

`experiments/runner.py` runs a deterministic synthetic task stream under four ablation modes (`none` / `cases` / `strategic` / `cost-aware`) and writes per-task metrics CSVs (quality, all five cost dimensions, cumulative cost, memory sizes) plus a `summary.json`. The C-vs-D separation — quality tied, cost divergent — is the experimental support for cost awareness being a necessary component of memory.

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/harness -p "test_*.py"
```

Pure stdlib; solver packages are optional extras discovered at runtime.

## License

MIT
