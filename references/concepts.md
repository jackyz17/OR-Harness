# Concepts: why OR-Harness is built this way

Read this page for the design rationale. Use [commands.md](commands.md) for calls, [modeling.md](modeling.md) for representations, [induction.md](induction.md) for knowledge formation, and [world_model_contract.md](world_model_contract.md) / [prediction_context.md](prediction_context.md) for prediction fields and input assembly.

Contents: [Research goal](#1-research-goal-and-core-variables) · [Problem structure](#2-understand-structure-before-selecting-a-strategy) · [Memory and retrieval](#3-learn-from-executions-and-retrieve-for-reuse) · [World model](#4-predict-consequences-to-support-decisions) · [Quality, cost, and risk](#5-preserve-quality-cost-and-risk-as-distinct-evidence) · [Offline evolution](#6-separate-online-solving-from-offline-evolution)

## 1. Research goal and core variables

OR-Harness studies whether an agent operating on a changing stream of optimization tasks can learn which strategies fit particular problem structures and achieve comparable or better solution quality at lower execution cost.

The difficult part is the interaction among decisions: shared capacity, cross-period state, alternative routes, logical rules, and other coupled constraints. A useful strategy must explain how it handles those interactions and under which conditions it remains applicable.

Use three core variables:

| Variable | Meaning |
|---|---|
| **P** | The current externally supplied OR problem and task context |
| **X = (Q, C, R)** | Outcome quality, execution cost, and risk/uncertainty |
| **H** | The Harness capability available to solve the problem |

The problem stream is externally driven. Changes in demand, capacity, or constraints produce new problems; they are not transitions caused by Harness actions. The world model predicts the consequences of a candidate strategy for solving the current problem.

Represent capability conceptually as `H = F(M, W, Pi, R, T)`: memory, world-model prediction, strategy/planning, retrieval/transfer, and tools/solvers. These sources interact. Counts of executions or entries provide indirect evidence about capability; they do not define a composite capability score. Here `R` denotes retrieval; the `R` in the outcome tuple denotes risk.

`H+` denotes an anticipated capability gain after completing a task and processing its experience. A predicted gain becomes supported only when the resulting knowledge or calibration improves later use. The gain is judged RELATIVE TO THE CURRENT HARNESS: a mature, well-known method can still yield a structural adaptation, a boundary it had not mapped, or a cost experience it lacked — but a new method or one more entry is not by itself a capability gain, and `none` is a reasoned no-gain judgement (`insufficient_basis` when the evidence cannot decide).

## 2. Understand structure before selecting a strategy

Profiling combines problem understanding with structural analysis. The Coupling-Aware Intermediate Representation (CIR) expresses entities, decisions, constraints, and their relationships before the agent chooses a strategy and builds the canonical optimization model.

This ordering lets structural understanding inform the modeling and solving approach. For example, a cross-period inventory balance may make preserving state essential when decomposing a planning problem. Discovering that relationship after choosing a decomposition would be too late to guide the choice.

Coupling scores summarize the CIR. Preserve the inspectable relationships as well as scalar summaries: the same score can arise from different mechanisms, and those mechanisms may require different methods. Constraint-variable co-occurrence supports a structural dependency; identifying shared resources or temporal state requires the corresponding semantics.

Freeze the task profile before strategy selection. Keep diagnostics from the subsequently built model and solver as execution observations. A strategy's implementation must not retrospectively change the structural identity under which it was selected.

Structural classes and quantized coupling cells organize conditional statistics. Their grouping rules belong in the representation and command references. The conceptual distinction is:

- **Statistical comparability:** whether observations may be pooled into a particular estimate.
- **Method applicability:** whether a technique's necessary premises hold for the current task.

Different cells may require separate estimates while still admitting transfer of a method after its premises are checked. Membership in one cell likewise does not establish that all methods observed there apply. Unknown structure records missing information; it does not establish similarity.

## 3. Learn from executions and retrieve for reuse

Two memory layers serve different purposes:

| Layer | Content | Role in future solving |
|---|---|---|
| **Execution Evidence** | Recorded methods, attempt chains, outputs, task checks, failures, and measured costs | Inspect how a previous task was handled and adapt grounded experience |
| **Strategic Knowledge** | Reusable methods, applicability conditions, explanations, boundaries, and supported effect claims | Guide what to do under specified conditions |

An execution record preserves observations and their sources. Its solver result can be valid for the implemented model while the answer remains wrong for the task. Record planned and actually performed methods separately; a plan supplies intent, while execution receipts and inspected artifacts support what happened.

A conditional statistic summarizes retained observations. Strategic knowledge adds an actionable commitment: a method, why it works, where it applies, and which effects the evidence supports. Expected quality or cost may accompany a strategy, but a method does not need a prediction interval to be useful.

Semantic induction is agent-led. The framework organizes execution material and computes declared checks; the agent compares relationships, explains operative steps, checks premises and boundaries, and decides whether to add, revise, merge, narrow, or produce nothing. A single execution can reveal a conditional method. Broader empirical advantage claims require comparable independent evidence. See [induction.md](induction.md) for support scopes and current publication rules.

### Retrieval and cold start

Embedding retrieval is the primary discovery channel; profile information assists organization and applicability assessment. Retrieve substantive method content, claims, and relevant task semantics rather than relying on method IDs.

Prefer admitted applicable knowledge as a starting point. When it is absent, inspect execution evidence directly; useful experience does not require prior promotion into strategic knowledge. Historical failures can reveal repair conditions or limitations even when the successful method is unsuitable.

A retrieved item keeps its evidence status. Text similarity exposes material to inspect; semantic premises determine whether its method can be adapted. Structural statistics retain their own scope: finding a related execution from another cell does not authorize adding its measured numbers to the target cell's estimate.

When memory offers no relevant experience, report that absence and reason from the current problem to propose a fresh strategy. The framework supplies no built-in method menu and no fabricated historical score for that proposal. Keep uncertainty explicit and obtain evidence through execution.

Task texts and embedding indexes support discovery within these two layers. They do not form another bank of outcome facts or strategic beliefs.

## 4. Predict consequences to support decisions

The world model provides a unified conditional forecast:

**`(P, H, candidate strategy) → (predicted X = Q/C/R, predicted H+)`**

A candidate specifies the proposed method and execution conditions. The predictor estimates their consequences: likely solution quality, required resources, failure risks, and potential learning. Solver/configuration choices are candidate inputs whose effects are assessed.

Use Q/C/R predictions alongside relevant evidence to compare candidates before execution. Historical experience describes observed cases; the forecast addresses the proposed action in the current context. The agent remains responsible for the choice, including uncertain predictions, quality requirements, and available budget.

The default comparison uses quality, cost, and risk. H+ explains what the action might teach and what would establish a later gain. It is not a measured reward or an automatic utility bonus. Adding an entry or predicting a benefit does not establish capability improvement.

A shadow configuration records forecasts without consuming them for selection. This is a diagnostic or ablation configuration; decision support is the world model's role in the main loop.

### Prediction and observation

Freeze the prediction input before the action. Include the relevant problem structure, consulted memory, current execution context, and predictor version; a historical replay must use the saved content rather than today's revised memory. Input construction itself produces no forecast.

Several unexecuted candidates may receive what-if predictions. Only an executed action supplies a real outcome for comparison. Bind forecasts to their actual executions and compare only defined predictions with corresponding observed dimensions. Missing receipts indicate unconfirmed conditions; known deviations require dimension-specific attribution under the contract.

Keep three forms of uncertainty distinct: missing evidence, the predictor's stated uncertainty, and reliability measured from resolved forecasts. Self-reported confidence is not a calibrated success probability. An unavailable predictor should remain distinguishable from a working service that returned no usable prediction.

Prediction calls also consume tokens and time. Their measured spend belongs in the actual execution ledger; the candidate's forecast cost remains a hypothesis. This makes the overhead of decision support visible in the final cost.

## 5. Preserve quality, cost, and risk as distinct evidence

| Dimension | Meaning | Evidence requirement |
|---|---|---|
| **Q: quality** | Task feasibility, correctness, and objective performance under the evaluation criteria | Model results plus task-level checks covering the relevant requirements |
| **C: cost** | Resources consumed by reasoning, tools, solving, checking, and rework | Measured values with provenance and compatible scope |
| **R: risk/uncertainty** | Failure exposure and uncertainty in quality, cost, or applicability | Observed failures, limited support, structural differences, and calibrated prediction reliability |

Three checks answer different questions:

1. **Model result:** did the solver solve the implemented optimization model?
2. **Task answer:** does the proposed solution satisfy the actual task and evaluation criteria?
3. **Knowledge claim:** do the stated method premises or empirical effect claims hold within their declared scope?

An optimal solver status does not validate omitted constraints or incorrect variable domains. A matching reference objective also does not establish every semantic requirement. Record task checks as passed, failed, or insufficient according to what was actually checked, with uncovered requirements explicit.

Retain confirmed-wrong attempts as failure evidence and include their measured cost. They must not continue to support a success claim. Repair and reflection can change the later answer while preserving the earlier observations.

Store cost as the raw vector:

**`llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`**

Preserve which dimensions were measured, their source, and their measurement scope. Missing cost is unknown. Distinguish estimates from observations, total tokens from partial token counts, and inner solver time from a subprocess wall-clock proxy. Do not silently average incompatible quantities or omit unmeasured records from a claim covering the whole cited set.

Compare attempt costs with attempt costs and complete task costs with complete task costs. The task total includes failed attempts, repairs, checks, and prediction overhead within its declared scope. Normalization and weighting are decision-policy operations; keeping raw dimensions permits later comparisons under different resource priorities.

Quality-preserving cost reduction is therefore a joint claim. Demonstrating lower solver runtime alone may leave total execution cost higher, and lower cost obtained by returning an invalid answer does not meet the research objective.

## 6. Separate online solving from offline evolution

| Phase | Responsibility |
|---|---|
| **Online, within an episode** | Profile the problem; retrieve knowledge/evidence; propose and compare strategies; make what-if predictions; execute; check answers; reflect or repair; record outcomes and cost |
| **Offline, after episode closeout** | Reconcile the full trajectory; assess eligible forecasts; publish calibration updates; review related evidence; induce and verify knowledge; revise the memory used by future tasks |

Within an episode, execution progress informs subsequent decisions while predictor parameters and learned calibration remain fixed. Recording an observation or binding a prediction is not an online model update. Calibrate and update the world model at episode boundaries, using the completed trajectory and real outcomes. An update may use calibration summaries and revised memory/context; weight training is not required by the design.

Knowledge evolution likewise occurs through explicit offline review. Match validation to the claim: inspect mathematical premises and solution preservation for method correctness, and compare measured executions for empirical effects. Framework verdicts cover declared computable checks; forward observations test subsequent reuse and refine confidence. These checks complement each other.

Revisions may strengthen, narrow, merge, or retire knowledge. A relevant counterexample challenges a claim whose premises held; a task outside its conditions indicates an applicability boundary. Structural-cell membership alone does not decide between those cases.

Keep evidence retention and belief revision separate. Complete old episode chains may leave a bounded evidence window after closeout processing. Expiry is a storage event, not a refutation; retained claims keep their content and verification scope. An expired reference cannot substitute for fresh support when revising a claim. Retirement and cold-archive policies prevent an unchanged failed pattern from repeatedly reappearing.

Evaluate H+ through later consequences: a derived method that is correctly reused, a calibrated forecast that improves selection, or a verified reduction in cost at comparable quality. Evidence accumulation, knowledge formation, and useful capability gain are separate events, potentially resolved at different times. A created or revised entry is a knowledge change, NOT an automatic capability improvement: it is recorded as unverified until a later use demonstrates the effect. An explicit `none` stance is still followed up — the operation's ACTUAL knowledge product is reported beside it as a fact, without treating that product as proof the "no gain" judgement was wrong. Pending future evidence is not a failed prediction.
