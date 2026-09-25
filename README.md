# OR-Harness

A small, composable, harness-friendly **strategy-learning capability layer** for operations-research agents.

> Given an evolving stream of large-scale industrial optimization tasks, can an OR agent learn from previous executions which solving strategies are appropriate for particular problem structures, and increasingly achieve similar or better solution quality at lower execution cost?

OR-Harness runs *inside* an outer harness agent (Hermes-style). It is not an autonomous agent: no conversation loop, no hidden global state, no unattended background work. The outer agent orchestrates; this layer advises, executes, and remembers.

A model is called **only** when you explicitly configure a provider (`--world-model URL::MODEL`, or `ORHarness(world_model=...)`) and explicitly invoke a prediction command. With no provider configured, every world-model command returns an explicit `not_configured` and no network activity happens. A prediction is a shadow hypothesis: it never becomes a fact, and an unexecuted candidate's prediction is never real feedback.

## What it provides

- **Two-layer memory**: an append-only **Execution Evidence Bank** (episodic facts: actual strategy/quality/cost, failures, artifacts; lossy compaction deferred) and a derived **Strategic Knowledge Bank** (commitments: expected quality/cost/failure risk, with prediction intervals and forward validation; admission never depends on evidence survival — re-inducible from retained evidence). Statistics are computed on the fly, never persisted.
- **Deterministic profiling**: structural coupling features from your task spec or supplied annotations — no NLP subsystem.
- **Strategy selection with full control, and NO built-in strategy directory**: transparent scoring (`α·Q̂ − β·C_scalar − γ·R̂`), two-layer evidence fallback, four ablation modes. There is no shipped list of method names, descriptions, applicability rules, actions or fallbacks. A strategy is a candidate because the MEMORY holds something about it in this structural cell — a recorded execution (`conditional_stats`) or an admission-verified claim (`strategic_entry`); with neither, `recall` returns an empty list plus `recommendations_basis` saying why, rather than a `no_memory` row at `-inf`. **You propose the candidates** (`recall --candidate`, `plan-next --candidates`), and `predict-cost`/`execute` accept ANY method id: an unknown cost basis is reported as `source="unknown"` with `expected_cost=null` (never a default zero), not a refusal. Entry content (`strategy_type` / `actions` / `fallback_strategy_id`) is what the harness recorded on the entry — induction fills in none of it, because inferring a method's actions from its id would be fabrication.
- **Sandboxed execution**: AST policy + POSIX rlimits + wall-clock timeout for your solve scripts; basic verification; five-dimensional cost metering (retries count).
- **Induction you control**: induction-pattern hints after every record (strategy contrast, intervention recovery, structural reproduction, advantage reversal); `induce` is always your explicit call. A claim's applicability is the family plus the structural cell its evidence occupies (legacy four-interval quantization — never a cross-sample span that would pool opposite regions); creating one needs ≥2 executions from ≥2 distinct tasks (repetition is not reproduction); publishing it needs a passed admission check (`induce --verify`: rule / repair / quality-preserving cost saving), so an unverified candidate is recorded but not recommended; cold archive with anti-resurrection.
- **Structured relation claims** (`induce --relation`): cross-task knowledge that is NOT one strategy's statistics — a structural condition paired with a modeling/solving choice and its consequence, e.g. "with temporal coupling ≥0.5 a temporal decomposition must keep its cross-period state". Each claim references recorded executions with a ROLE and declares what is computationally checkable (`probe` / `status` / `comparison` assertions); the framework derives the evidence identity from the facts and computes the verdict itself. Assertions carry their own semantics: `mode: paired` compares only same-task pairs, `mode: group` compares side means over a metric every record measured, and `aggregation: all|mean` decides how an unfavourable sample is treated. A verified claim means "no violation found within this scope" — the scope (executions, tasks, roles, which assertions were checked) travels with it. Publication is **per relation**, on its own verdict plus ≥2 independent tasks, so it neither publishes nor is granted by the host entry's statistical admission, and a single-task repair stays a verified fact about that task. `recall` carries published relations in a separate `knowledge` section (with their state and `newer_evidence_since_verification`), so an unverified or refuted claim is never dressed up as available knowledge. See [references/induction.md](references/induction.md).
- **Seven solver adapters** (highs, pulp, ortools, scip, copt, pyomo, gurobi) — availability probing only; you pick the concrete solver per situation.
- **Unified world-model contracts** (`wm-contract/1`): a versioned, serializable, validatable shape for two prediction modules — **OR strategy consequence prediction** (benefit with its metric/unit/baseline, `CostVector` cost, named risk events, uncertainty split into execution randomness vs evidence gap) and **harness capability evolution prediction** (capability evidence for `H = F(M, W_OR, Pi, R, T)`, a candidate learning operation, baseline and horizon, learning cost, degradation risk, verification conditions). Building a contract makes no model call, so it says `status="contract_only"` rather than pretending a forecast was made, and it keeps `provider_configured` (a provider is attached), `service_available` (this build implements that kind) and `prediction_made` (a forecast really happened) as three separate facts: a configured provider with zero model calls is never `valid`. Legacy unversioned payloads stay readable through an explicit legacy view; an unknown contract version fails instead of being guessed at. See [references/world_model_contract.md](references/world_model_contract.md).
- **Frozen prediction input context** (`wm-context/1`): what a prediction is actually conditioned on, and how that information reaches the model consistently, completely and traceably. One `orx context` freezes the **joint problem representation** (task text and payload, the CIR's relations kept as relations rather than compressed into three numbers, math attributes with an explicit origin each, the structural profile and its derivation report), **X/B** from the same snapshot, the **retrieval evidence of both existing channels** (deduplicated by evidence identity, carrying content and applicability labels — a cross-cell hit stays visible and never enters the target cell's statistics), the **harness capability evidence** (`H = F(M, W_OR, Pi, R, T)`, evidence strength only, no composite score) and the **external execution constraints**. It builds BEFORE modelling: no `model`, no CIR and no task text are all normal inputs whose absent parts are reported item by item, and nothing is inferred from the `family` name. Building performs **no prediction-model call, no solver execution and no induction**; one context is shared across a candidate comparison, and reuse is version-verified. See [references/prediction_context.md](references/prediction_context.md).
- **Strategy-outcome prediction service** (`wm-so/1`): a training-free OR strategy-consequence prediction service over the frozen context. `orx predict-strategy` predicts ONE candidate's benefit (with metric/unit/baseline — a `solution_quality` value is normalized, a raw objective is refused), resource cost (a `CostVector` whose mask marks predicted dimensions), risk (named events, separate from cost) and uncertainty (a model self-report recorded as explicitly uncalibrated); `orx plan-next` compares candidates on one conservative yardstick (unknown cost charged the peak share, unknown risk the full weight, unknown benefit nothing — decision rules, not measured probabilities) and suggests; `orx bind-strategy` links the real execution to the prediction of the configuration actually proposed, with identity checks where an UNKNOWN identity field is recorded separately and never counts as a match. Every failure state (not configured, provider error, empty/invalid payload, out-of-range values) is distinguishable and persisted with the call's real cost. Planning speaks THIS protocol only (there is no second prediction path): the legacy `predict_outcome` Python API is kept for reading old records, but no CLI/agent flow uses it. See [references/strategy_outcome.md](references/strategy_outcome.md).
- **Episode close-out and experience calibration**: `orx close-episode` closes one episode under an honest terminal state (completed/failed/aborted/budget_exhausted), evaluates every bound strategy-outcome prediction against its real outcome **field by field** (benefit error under the prediction's own declared metric/baseline, per-dimension cost error where both sides measured the same scope, Brier scores for labelled risk events, interval coverage — excluded fields are neither hits nor misses), and publishes a versioned **experience calibration summary** that later episodes' prediction contexts read (closed episodes only; below the sample minimum it reports `insufficient_evidence`, never a guessed figure). Idempotent; no solver run, no model call, no induction; the frozen prediction is never rewritten and an unexecuted candidate never gets a counterfactual label. The same strategy chosen twice is two selection rounds (`round_index`), never one aggregated sample. See [references/episode_closeout.md](references/episode_closeout.md).
- **Capability-evolution prediction and two-stage effect feedback** (`wm-ce/1`): `orx predict-capability` predicts what ONE offline learning operation (induce / revise / reverify / retire) would change in FUTURE task performance — expected changes with their metric/unit/baseline, learning cost, degradation risk, uncertainty and verification conditions — from frozen capability evidence, a candidate operation, the exact experience scope and a framework-frozen per-metric baseline. `orx compare-capability` applies ONE bounded rule (largest net saving over the declared window: the cumulative saving minus the one-time predicted maintenance cost, in the SAME unit, under a quality-non-degradation constraint) and reports rather than ranks incomparable candidates. `orx accept-capability` is the only offline entry that changes knowledge and runs the operation the prediction was about; `orx bind-capability` records the maintenance FACT; `orx evaluate-capability` judges the EFFECT against real later tasks — counted in TASK-EPISODES, admitting only tasks whose WORK ran after the operation, and only once the declared horizon is met. A knowledge change is never a capability gain. See [references/commands.md](references/commands.md).

## Quick start

```bash
pip install -e .                # zero runtime dependencies (pure stdlib)
pip install -e ".[solvers-free]"  # optional: highspy + pulp

orx doctor                       # probe solvers, check memory home
orx profile   --task t.json
orx recall    --task t.json --top 3 --candidate my-method --candidate alt-method
orx execute   --task t.json --strategy my-method --code solve.py --workspace ws --solver highs
orx check-task exec_... --check '{"reference_objective": 10755, "integer": {"variables": ["x1", "x2"]}}'
orx record    --execution exec.json --override llm_tokens=1840
orx induce    --strategy my-method
orx inspect   --bank strategic
```

`my-method` is deliberately a name the framework has never seen: it ships no strategy directory, so there is nothing to be a member of. `recall` reports what memory holds for the methods you name (initially: nothing, with a reason), and `predict-cost`/`execute` run any method you propose. The full walkthrough — empty bank, proposal, unknown cost, execution, task check, record, recall — is a runnable script: `PYTHONPATH=src python3 references/examples/no_catalog.py`.

Memory lives in an explicit directory (`--home` or `$OR_HARNESS_HOME`), stored as a single SQLite file.

Configuration is deliberately explicit — there is no config file; the whole surface is CLI flags plus environment variables: `$OR_HARNESS_HOME` (memory location), `$OR_WM_API_KEY` (world-model endpoint key; URL and model are passed via `--world-model URL::MODEL`), and `$OR_EMBEDDING_BACKEND` / `$OR_EMBEDDING_BASE_URL` / `$OR_EMBEDDING_MODEL` / `$OR_EMBEDDING_API_KEY` (embedding backend for vector recall). Everything else is per-invocation flags — see [references/commands.md](references/commands.md).

## Documentation

- **[SKILL.md](SKILL.md)** — the thin contract for harness agents (start here)
- **[references/world_model_contract.md](references/world_model_contract.md)** — the unified prediction contracts, `contract_only`, attempt vs strategy window, capability sources, migration table (with a runnable example)
- **[references/prediction_context.md](references/prediction_context.md)** — the frozen prediction input context: the joint representation and its math-attribute origins, the two retrieval channels and their evidence classes, capability evidence strength, one-build/consistent-reuse rules (with a runnable example)
- **[references/strategy_outcome.md](references/strategy_outcome.md)** — the strategy-outcome prediction service (wm-so/1): the protocol, the comparison yardstick, the prediction–choice–execution binding (with a runnable example)
- **[references/episode_closeout.md](references/episode_closeout.md)** — the episode close-out: real outcome summaries, per-field post-hoc evaluation, window selection rounds, the experience calibration channel (with a runnable example)
- **[references/concepts.md](references/concepts.md)** — two-layer memory, CostVector, disposal ladder, and the three separate questions (did the solver solve the model / is the answer valid for the task / does the claim hold)
- **[references/induction.md](references/induction.md)** — the four induction-worthy patterns, applicability as family + structural cell, creation gate and admission verification, offline lifecycle
- **[references/examples/task_check.py](references/examples/task_check.py)** — runnable: check → diagnose → repair → record → close-out (a relaxed LP is caught by an integer-domain check, repaired, and only then treated as a success)
- **[references/examples/](references/examples/)** — the other runnable walkthroughs (`episode_closeout.py`, `strategy_outcome.py`, `contract_roundtrip.py`, `prediction_context.py`, `capability_evolution.py`)

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
