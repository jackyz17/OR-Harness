# Host operations and execution diagnostics

For runners, integrations and exceptional recovery. Ordinary solving uses [commands.md](commands.md); load the relevant section only when reporting or diagnosing these facts.

## Submission and runtime diagnostics

Existing execution statuses stay `error`, `timeout` or the script's result
status. Read `execution.execution_features.technical_outcome`:

| Case | `failure_stage` | `process_started` / `executed` | Risk evidence |
|---|---|---|---|
| Syntax or documented script-contract rejection | `submission` | false / false | implementation failure |
| Missing, changed or invalid input | `submission` | false / false | cause Unknown |
| Process cannot launch | `launch` | false / false | environment failure |
| Required isolation setup fails | `sandbox_setup` | true / false | environment failure |
| Python/API exception or missing dependency | `runtime` | true / true | code error → implementation; dependency/license unavailable → environment |
| Invalid delivered JSON/protocol | `result` | true / true | implementation failure |
| Harness interrupts script | `runtime`, status `timeout` | true / true | formal timeout; code cause remains Unknown without evidence |
| Harness deadline interrupts isolation setup | `sandbox_setup`, status `timeout` | true / false | formal timeout; algorithm runtime unobserved |
| Solver's own time limit returns readable result | no failure stage | true / true | no Harness timeout |

The receipt also keeps `failure_cause`, `submission_rejected`, `result_received`
and `runtime_eligible`. A rejected attempt has formal timeout=0 because it was
not interrupted, but `runtime_eligible=false`: it is no algorithm runtime
evidence. Formal timeout labels/Brier retain their existing definition;
`technical_r.runtime_coverage` reports this distinction separately. Actual
submission/launch latency is recorded; solver runtime stays unmeasured before
script startup. Prediction, action, script, input refs, reason and spent cost
survive failure. Record it, then obtain a fresh plan-next before a repaired
submission; no implicit solve retry is introduced. Historical receipts and
probabilities are not rewritten by this diagnosis.

Use the numbers and method steps from the actual run. Stamp `config` and `method_performed` with the current `OR_ACTION_ID`. Read performed steps in `method_actual` and the plan in `method_planned`. `execution_config` identifies executor observations, script read-back and configured settings. Executor observations take precedence. `script_timeout_s` gives the whole-script limit.

Use `--adapted-from` and `--adaptation` to cite past cases. Use `--used-entry-ids 3@2,5@1` to declare versions actually adopted. Pass `--used-entry-ids ''` to report no adoption.

**Read and continue** Read `result.execution_id`, `action_id`, `execution` and `prediction_binding`. `solution_variables` retains up to 2000 variables. For an exit-2 precondition error, check the saved P and task identity. Run `check-task` and record every real attempt, including a failed one. Use a new `plan-next` selection for a repaired attempt.

## Record and host usage

### `orx record`

```text
orx record (--execution JSON|PATH | --from-staged ID)
  [--usage-file JSON|PATH] [--usage-host HOST]
  [--override llm_tokens=1840,tool_calls=9] [--override-mode replace|increment]
  [--override-source agent_estimate|agent_observed|provider_usage] [--force]
  [--prediction JSON] [--method JSON] [--method-actual JSON]
  [--used-entry-ids ENTRY@VERSION]
orx record --discard-staged ID
orx record --session --task T --reason TEXT [--episode E] [--cancelled]
  [--attempted-execution EX_ID] [--override ...]
```

Use `--from-staged EX` to save the original attempt, including its check and receipts. `--execution` imports a supplied record. `--discard-staged` drops a pending item. `--method`, `--method-actual` and `--used-entry-ids` fill missing plan, performed-method and adoption fields.

Read `recorded`, `prediction_checks[]`, `cost_completeness`, `cost_feedback` and `index_sync`. Quality feedback uses the versions declared adopted. An entry's declared prediction supports a hit/miss. A recorded observation without a prediction has `hit=null`. Attach available usage and save attempts with incomplete costs. Use `amend-cost` for a late report. Use `index_sync` and `unrecorded_staged_executions[]` to find follow-up work.

**Host stop** Use `--session` once the host confirms termination or cancellation. Keep the host's stated reason. Attach `--attempted-execution EX` when a staged attempt exists. A running action with no execution fact is recorded as interrupted. With no attempt, the command records `finish_task` for close-out. See [host stops and interrupted runs](strategy_outcome.md#4-budgets-stops-and-archiving-an-interrupted-run).

### Usage reports and overrides

The executor records available latency and solver runtime. Pass whole-attempt host usage with `--usage-file` as inline JSON or a path.

```json
{"schema":"or-host-usage/1","host":"generic","model":"your-model",
 "scope":{"task_id":"t1","episode_id":"ep1","attempt_id":"ex_..."},
 "tokens":{"prompt_tokens":700,"completion_tokens":300,
           "reasoning_tokens":120,"cached_tokens":40,"calls":3},
 "tool_calls":9,"tool_calls_lower_bound":1,
 "measured":["prompt_tokens","completion_tokens","tool_calls"],
 "provenance":{"llm_tokens":"provider_usage","tool_calls":"agent_observed"},
 "notes":"whole-attempt host report"}
```

| Item | Rule |
|---|---|
| Token totals | Prompt plus completion. Reasoning and cached tokens are breakdowns |
| Partial token totals | One observed side is a lower bound with `prompt_only` or `completion_only` |
| Measured dimensions | Use the report's `measured` list and observed values |
| Tool calls | Report all calls in scope, at or above the sandbox lower bound |
| Report plus override | Use overrides to fill different dimensions |
| Override provenance | Bare overrides are `agent_estimate`. Use `agent_observed` for observed reports and `provider_usage` for provider measurements |

`--usage-host openclaw` looks for `<home>/host_usage/openclaw-usage.<execution_id>.json`, then `openclaw-usage.json`. `generic` uses `host-usage.json`. `OR_HOST_USAGE_FILE` is the fallback. Missing reports leave the affected measurements unknown.

Reports can include window, counter or call evidence for [C observations](episode_closeout.md#c-observations). A wider scope is stored alongside attempt costs. Submit later reports through `amend-cost --usage-file`.

## Cost corrections

### `orx amend-cost <execution_id> [--override llm_tokens=1840,tool_calls=9] [--usage-file JSON|PATH] [--usage-host HOST] [--usage-source host] [--mode replace|increment] [--source agent_observed|agent_estimate|provider_usage] [--force]`

Update recorded costs for EX. `replace` is the default. Use `increment` for an additional measured amount in the same scope. Use `--source` to state provenance and `--force` to replace framework measurements. Read `cost`, `cost_measured`, `still_missing` and normalized `usage` when supplied.

## Action reports

### `orx action --report TYPE --task t.json [--episode ep1] [--params JSON] [--outcome JSON] [--cost JSON] | --amend-cost ACTION_ID --cost JSON`

Record an action performed outside `execute`. Choose `model|select_strategy|verify|finish_task` and supply observed parameters, outcome and cost. Supplied cost dimensions are marked measured. Read the action with `inspect --bank actions`. Use `--amend-cost` to replace its cost.

`--task` requires a task JSON object or JSON file containing `task_id`, not a bare task ID. `--params`, `--outcome` and `--cost` also require JSON objects or JSON files; malformed input returns a named argument error. `action --report verify` records an agent-reported action and does not submit G attribution. Use only `check-task EX --check check.json` for that; putting `g_attribution` in action params/outcome is rejected with the correct interface.

### Implementation and correction reports

The host uses the existing `orx action --report TYPE --task t.json --episode E --outcome report.json` (or Python `report_action` / `end_action`). Report facts from the actual solving workflow; batch concise facts when convenient. `plan-next` establishes the selection anchor and `execute` reuses it. The report names the selected prediction P.

| Outcome field | When to submit | Required content |
|---|---|---|
| `implementation_facts` | When a real defect or corrective round is evidenced | List of facts with prediction, stable event ID, actual occurrence time, phase, classification, observer and evidence references |
| `implementation_coverage` | At candidate completion or before moving to the next decision | Prediction P, actual `through`, `implemented`, `complete`, `candidate_complete`, observer and evidence references; `from` defaults to selection time |
| `correction_coverage` | With `finish_task` | `from` = earliest covered decision start, `through` = actual task-end timestamp, `complete`, observer and evidence references |

Example `model` outcome for an observed implementation defect and a started repair (replace identifiers, timestamps and references with actual observations):

```json
{
  "implementation_facts": [{
    "prediction_id": "P",
    "event_id": "defect-17",
    "occurred_at": 1795651201.0,
    "phase": "pre_execute",
    "classification": "implementation",
    "confirmed": true,
    "requires_correction": true,
    "observer": "runner-session-42",
    "summary": "Expression constructor rejected a native scalar.",
    "evidence_refs": ["artifact:failure.py", "host-log:42/test17"],
    "correction": {
      "round_id": "repair-8",
      "started_at": 1795651202.0,
      "method": "Normalize scalar types before constructing expressions.",
      "evidence_refs": ["artifact:repair.diff"],
      "result": "targeted check passed",
      "verification_refs": ["host-log:42/test18"]
    }
  }]
}
```

Phases: `implementation`, `pre_execute`, `execution`, `technical_verification`. Classifications: `implementation`, `environment`, `framework`, `acceptance`, `mathematical`, `other`, `unknown`. Use `obstructed: true` for an evidenced obstacle; omit `correction` if no repair started. Mark routine probes or expected handling `excluded: true` if reported. Record actual root cause and verification outcome, or Unknown. Each report accepts up to 32 facts, 6,000 characters per fact; retain full artifacts in referenced evidence.

For an error inside formal execution, supply `execution_id: EX`. The terminal formal implementation error uses the canonical `execution:EX:technical` identity. Distinct recovered defects in the same execution retain distinct host event IDs; set `formal_error: false` when the execution also has a terminal error. A later formal resubmission belonging to an existing repair is associated by re-reporting the same fact and round with its original times and `correction.retry_action_id` set to the real retry execute action ID.

Example `verify` outcome at candidate completion:

```json
{"implementation_coverage":{"prediction_id":"P","through":1795651210.0,
 "implemented":true,"complete":true,"candidate_complete":true,
 "observer":"runner-session-42","evidence_refs":["host-log:42/implementation-span"]}}
```

Example `finish_task` outcome (use the same actual terminal time as the finish action):

```json
{"finished":true,"correction_coverage":{"from":1795651200.0,"through":1795651220.0,
 "complete":true,"observer":"runner-session-42","evidence_refs":["host-log:42/correction-span"]}}
```

In Python, set `started_at` / `ended_at` on `report_action` when supplying measured action boundaries. Use `complete: false` when coverage is incomplete. See [observation and deduplication rules](episode_closeout.md#implementation-observation-and-correction-counting) for evaluation. Formal execution quotas use their own attempt increments.

## Budget and retry observations

### `orx budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]`

Read episode usage or save limits with `--declare`. Tokens, tool calls, solver runtime and retries accumulate over the episode. Latency limits apply per attempt. Read `status`, `exceeded_dims`, `episode_exhausted` and `attempt_limited`. Unknown usage for a declared dimension leaves the budget `unconfirmed`. An exhausted episode or confirmed host stop blocks new planning and execution.

`execute` automatically observes each real attempt's retry increment within
the task/episode, including failures and changes of strategy or solver.
Three attempts record 0/1/1 and consume 2 retries. The corresponding
current world-model `retries` predicts started corrective/retry rounds from
its own decision through task termination (`wm-retry-corrective-rounds/1`).
It needs separate coverage and must not be backfilled as a formal increment.
See [implementation and correction reports](#implementation-and-correction-reports).
