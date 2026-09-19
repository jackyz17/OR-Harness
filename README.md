# OR-Harness

A small, composable, harness-friendly **strategy-learning capability layer** for operations-research agents.

> Given an evolving stream of large-scale industrial optimization tasks, can an OR agent learn from previous executions which solving strategies are appropriate for particular problem structures, and increasingly achieve similar or better solution quality at lower execution cost?

OR-Harness runs *inside* an outer harness agent (Hermes-style). It is not an autonomous agent: no conversation loop, no hidden global state, no unattended background work. The outer agent orchestrates; this layer advises, executes, and remembers.

A model is called **only** when you explicitly configure a provider (`--world-model URL::MODEL`, or `ORHarness(world_model=...)`) and explicitly invoke a prediction command. With no provider configured, every world-model command returns an explicit `not_configured` and no network activity happens. A prediction is a shadow hypothesis: it never becomes a fact, and an unexecuted candidate's prediction is never real feedback.

## What it provides

- **Two-layer memory**: an append-only **Execution Evidence Bank** (episodic facts: actual strategy/quality/cost, failures, artifacts; lossy compaction deferred) and a derived **Strategic Knowledge Bank** (commitments: expected quality/cost/failure risk, with prediction intervals and forward validation; admission never depends on evidence survival — re-inducible from retained evidence). Statistics are computed on the fly, never persisted.
- **Deterministic profiling**: structural coupling features from your task spec or supplied annotations — no NLP subsystem.
- **Strategy selection with full control**: transparent scoring (`α·Q̂ − β·C_scalar − γ·R̂`), two-layer evidence fallback, four ablation modes.
- **Sandboxed execution**: AST policy + POSIX rlimits + wall-clock timeout for your solve scripts; basic verification; five-dimensional cost metering (retries count).
- **Induction you control**: C1–C6 evidence hints after every record; `induce` is always your explicit call. A claim's applicability is the family plus the structural cell its evidence occupies (legacy four-interval quantization — never a cross-sample span that would pool opposite regions); creating one needs ≥2 executions from ≥2 distinct tasks (repetition is not reproduction); publishing it needs a passed admission check (`induce --verify`: rule / repair / quality-preserving cost saving), so an unverified candidate is recorded but not recommended; cold archive with anti-resurrection.
- **Seven solver adapters** (highs, pulp, ortools, scip, copt, pyomo, gurobi) — availability probing only; you pick the concrete solver per situation.
- **Unified world-model contracts** (`wm-contract/1`): a versioned, serializable, validatable shape for two prediction modules — **OR strategy consequence prediction** (benefit with its metric/unit/baseline, `CostVector` cost, named risk events, uncertainty split into execution randomness vs evidence gap) and **harness capability evolution prediction** (capability evidence for `H = F(M, W_OR, Pi, R, T)`, a candidate learning operation, baseline and horizon, learning cost, degradation risk, verification conditions). The CONTRACT is implemented; the prediction SERVICE is not attached — a built contract says `status="contract_only"` rather than pretending a forecast was made, and it keeps `provider_configured` (a provider is attached), `service_available` (this build implements that kind) and `prediction_made` (a forecast really happened) as three separate facts: a configured provider with zero model calls is never `valid`. Legacy unversioned payloads stay readable through an explicit legacy view; an unknown contract version fails instead of being guessed at. See [references/world_model_contract.md](references/world_model_contract.md).
- **Frozen prediction input context** (`wm-context/1`): what a prediction is actually conditioned on, and how that information reaches the model consistently, completely and traceably. One `orx context` freezes the **joint problem representation** (task text and payload, the CIR's relations kept as relations rather than compressed into three numbers, math attributes with an explicit origin each, the structural profile and its derivation report), **X/B** from the same snapshot, the **retrieval evidence of both existing channels** (deduplicated by evidence identity, carrying content and applicability labels — a cross-cell hit stays visible and never enters the target cell's statistics), the **harness capability evidence** (`H = F(M, W_OR, Pi, R, T)`, evidence strength only, no composite score) and the **external execution constraints**. It builds BEFORE modelling: no `model`, no CIR and no task text are all normal inputs whose absent parts are reported item by item, and nothing is inferred from the `family` name. Building performs **no prediction-model call, no solver execution and no induction**; one context is shared across a candidate comparison, and reuse is version-verified. See [references/prediction_context.md](references/prediction_context.md).
- **Strategy-outcome prediction service** (`wm-so/1`, world-model M3): a training-free OR strategy-consequence prediction service over the frozen context. `orx predict-strategy` predicts ONE candidate's benefit (with metric/unit/baseline — a `solution_quality` value is normalized, a raw objective is refused), resource cost (a `CostVector` whose mask marks predicted dimensions), risk (named events, separate from cost) and uncertainty (a model self-report recorded as explicitly uncalibrated); `orx plan-next --protocol strategy-outcome` compares candidates on one conservative yardstick (unknown cost charged the peak share, unknown risk the full weight, unknown benefit nothing — decision rules, not measured probabilities) and suggests; `orx bind-strategy` links the real execution to the prediction of the configuration actually proposed, with identity checks and honest mismatch recording. Every failure state (not configured, provider error, empty/invalid payload, out-of-range values) is distinguishable and persisted with the call's real cost. The legacy `predict_outcome` path is unchanged. See [references/strategy_outcome.md](references/strategy_outcome.md).

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
- **[references/world_model_contract.md](references/world_model_contract.md)** — the unified prediction contracts, `contract_only`, attempt vs strategy window, capability sources, migration table (with a runnable example)
- **[references/prediction_context.md](references/prediction_context.md)** — the frozen prediction input context: the joint representation and its math-attribute origins, the two retrieval channels and their evidence classes, capability evidence strength, one-build/consistent-reuse rules (with a runnable example)
- **[references/strategy_outcome.md](references/strategy_outcome.md)** — the strategy-outcome prediction service (wm-so/1): the protocol, the comparison yardstick, the prediction–choice–execution binding (with a runnable example)
- **[references/concepts.md](references/concepts.md)** — two-layer memory, CostVector, disposal ladder
- **[references/induction.md](references/induction.md)** — C1–C6, applicability as family + structural cell, creation gate and admission verification, offline lifecycle
- **[references/examples.md](references/examples.md)** — four complete walkthroughs

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
