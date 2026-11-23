# OR-Harness concepts

OR-Harness uses task structure, execution evidence, and strategy predictions to guide solving and learning. See the [command reference](commands.md), [modeling guide](modeling.md), [world-model contract](world_model_contract.md), and [induction guide](induction.md).

[Research goal](#1-research-goal-and-core-variables) · [Problem structure](#2-understand-structure-before-selecting-a-strategy) · [Memory and retrieval](#3-learn-from-executions-and-retrieve-for-reuse) · [World model](#4-predict-consequences-to-support-decisions) · [Quality, cost, and risk](#5-preserve-quality-cost-and-risk-as-distinct-evidence) · [Offline evolution](#6-separate-online-solving-from-offline-evolution)

## 1. Research goal and core variables

OR-Harness studies how an agent can adapt its strategies across a stream of optimization problems while preserving or improving solution quality and reducing execution cost. Shared capacity, cross-period state, alternative routes, and logical rules determine which strategies fit a task.

| Variable | Meaning |
|---|---|
| **P** | The current externally supplied problem and its task context |
| **X = (Q, C, R)** | Solution quality, execution cost, and one-attempt technical risk |
| **H** | The capability available through memory, prediction, planning, retrieval, and tools |

Changes in demand, capacity, or constraints arrive as new tasks. For the current task, the world model forecasts the consequences of the agent's candidate strategies.

## 2. Understand structure before selecting a strategy

The Coupling-Aware Intermediate Representation (CIR) describes the entities, decisions, constraints, and relationships that guide modeling and solving. For example, an inventory balance connects adjacent periods. A decomposition that respects that connection carries the needed inventory state across periods.

Coupling scores summarize the CIR. Inspect the relationships behind the scores to identify the mechanism. Variables sharing a constraint indicate a dependency. Constraint semantics identify whether that dependency comes from a shared resource, temporal state, or another source.

Freeze the task profile before selection so the forecast and the decision refer to the same problem structure. Store later modeling and solver diagnostics with the execution.

Structural classes and coupling cells group observations for conditional estimates. Pool measurements within a comparable scope. Assess whether a method transfers by checking its premises on the new task, including tasks in other cells.

## 3. Learn from executions and retrieve for reuse

| Memory layer | Content | Use |
|---|---|---|
| **Execution Evidence** | Actual methods, attempt chains, outputs, checks, failures, and measured costs | Examine how a previous attempt worked, failed, or was repaired |
| **Strategic Knowledge** | Reusable **when / do / expect** rule, with optional machine-checkable structural predicates and recorded verification | Read **when** and decide whether to adopt; a retrieval hit or a machine-compatible cell is not adoption |

Keep the proposed method with its prediction and the performed method with the execution receipt and artifacts. A new rule version records **when / do / expect**. The agent uses evidence to decide what to record, whether to revise an existing rule, and whether to keep a candidate or publish it within a supported scope. A sound derivation or a small executable check can support a scoped rule during cold start. Adoption links a specific version to an execution. [Induction and rule use](induction.md)

### Retrieval and cold start

Embedding retrieval finds related methods and executions. The task profile helps organize them. Check the original constraints and the rule's premises to decide what to adapt. A failed attempt can reveal a repair or a boundary, and a retrieved method carries the scope of its existing evidence.

When relevant memory is sparse, propose a method from the task itself. The resulting execution provides evidence for later retrieval and review.

## 4. Predict consequences to support decisions

**`(P, H, candidate strategy) → (G, C, R, H+)`**

A candidate describes the proposed method and its execution conditions, including the solver and settings. **Q** describes quality on the original task. **G** is the benefit forecast used by `plan-next` to score a proposal. By default it estimates the probability of original-task acceptance given faithful execution. **C** forecasts resources over a declared scope. **R** forecasts the probability of technical implementation failure in one attempt. Utility uses G, normalized C, and R on a comparable basis. [Scored selection](strategy_outcome.md)

**H+** is the pre-execution learning decision relative to the rule library available at prediction time. Its hypothesis names what reusable method or boundary the execution could reveal. During H+ building, a bound execution with `H+=1` schedules one review at close-out. An executed `H+=0` skips that review. [H+ review dispatch](induction.md#when-to-start-induction)

### Prediction and observation

Freeze the problem profile, retrieved memory, execution context, and predictor version for the decision. Forecast each candidate in that context. Bind the chosen forecast to its actual execution and compare corresponding observations using the forecast's declared scope.

Record missing evidence, forecast uncertainty, and reliability measured from resolved forecasts in their respective fields. Direct technical failure observations support the R estimate. Record the resources spent on prediction calls in the measured ledger.

## 5. Preserve quality, cost, and risk as distinct evidence

| Dimension | What to observe |
|---|---|
| **Q · solution quality** | Feasibility, correctness, and objective performance under the original task's acceptance criteria. Record which requirements the task check covers. |
| **C · execution cost** | Measured reasoning, tool, solver, checking, and repair resources, each with its source and measurement scope. |
| **R · technical risk** | One-attempt evidence for three technical hazards: code/API path or result delivery (`implementation_failure`), an execution-timeout interruption before a usable result (`timeout`, distinct from a solver that returns a usable result at its own limit), and environment blockage such as dependencies, licences, permissions or external services (`environment_failure`). Record verified failure or valid delivery; the U risk term is the maximum of the events every ranked candidate supplied, not a joint failure probability. Unverified causes stay unknown. |

The solver status describes the model it solved. The original-task check examines the actual requirements, including domain and business constraints. A knowledge review tests the rule's premises or measured effect within the stated scope. Keep failed attempts and their measured costs as evidence for repair and comparison.

Record the raw cost dimensions.

**`llm_tokens, output_tokens, tool_calls, solver_runtime_s, retries, latency_s, remaining_latency_s`**

An execution's `retries` is its own 0/1 increment in the task/episode chain;
three attempts record 0/1/1 and total 2. A remaining-cost forecast predicts
additional attempts after its bound baseline, whose actuals for those three
decisions are 2/1/0. Strategy and solver changes do not reset the chain.

Compare costs over matching scopes. Include failed attempts, checks, repairs, and prediction calls in the task measurement where they fall within its declared scope. Record missing dimensions as unknown and retain the source of measured values and proxies. Assess quality-preserving cost reductions with both original-task checks and comparable measurements.

## 6. Separate online solving from offline evolution

| Phase | Responsibility |
|---|---|
| **Online solving** | Build the profile, retrieve evidence, score candidates, execute, check and repair as needed, then commit the answer. |
| **Episode close-out** | `close-episode` evaluates eligible bound forecasts against the completed trajectory, publishes calibration, and returns any required H+ review IDs. |
| **Offline induction** | Take the returned reviews and compare execution evidence. Record reusable findings, revise rules, or submit a conclusion about the evidence. |

Facts from an attempt can guide later decisions within the same episode. Later planning uses the available calibration and learning receipts with the rule versions in its new context. [Episode close-out](episode_closeout.md) · [Learning receipts](induction.md#read-publication-completion-and-learning-separately)

When extending or narrowing a rule, test its premises, preservation of the original task result, or measured effects as appropriate. Later independent use can strengthen its scope. The evidence window bounds raw execution history. Keep each published rule's scope with its supporting record.
