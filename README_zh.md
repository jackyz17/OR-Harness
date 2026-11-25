# OR-Harness

A composable strategy-learning layer for operations-research agents. The outer agent proposes methods, writes solver code, judges evidence and forms reusable rules. OR-Harness supplies structural profiling, memory retrieval, G/C/R/H+ prediction, U-scored decisions, sandboxed execution, task checking, versioned knowledge and factual feedback.

The research question is whether repeated, attributable executions can teach an OR agent which methods fit a problem structure while preserving task quality and reducing resource use. Scripted examples verify the workflow; they do not establish predictive discrimination or a measured cost advantage.

## Start here

Install from this checkout:

```bash
python3 -m pip install -e .
```

Python 3.9+ and the standard library support the core; solver packages are optional and availability is checked at runtime. Configure an OpenAI-compatible world-model endpoint/model (`--world-model URL::MODEL`, credential `OR_WM_API_KEY` when required). Formal execution needs a permitted U-scored selection, including for one candidate. Without a usable predictor, planning reports why it cannot select an executable prediction.

正式执行需要 Linux 用户、挂载和网络命名空间、`unshare` 及 `libseccomp.so.2`。
`orx doctor` 检查隔离能力；隔离不可用时拒绝运行。文件任务通过 task JSON 的
`input_resources` 声明输入，脚本在隔离目录的 `inputs/<path>` 读取原始文件。
参见[文件输入与安全边界](references/commands.md#file-input-and-isolation)。

Read [SKILL.md](SKILL.md) to complete one task in six steps:

| Step | Action | Main result |
|---|---|---|
| 1 | Prepare/reuse `task.json`; `profile` and `recall` | Structure, CIR health, applicable rules and real cases |
| 2 | `plan-next --task task.json --episode EP --candidates candidates.json` | `selected_prediction_id`, `selected_score`, per-candidate reasons |
| 3 | Execute, check the original task, and record each attempt | Execution result, check coverage, and measured resources |
| 4 | Commit the answer and run `close-episode --task T --episode EP --terminal ...` | Published calibration and `induction_schedule.pending_ids` |
| 5 | `induction-material --induction-id J`, then one review submission | Rule/publication choice and an actual review receipt |
| 6 | Adopt `ENTRY@VERSION`, check/record/close, then plan the next task | Versioned use, optional utility management and relevant feedback |

Use the same task file and memory home for a problem; put a shared CIR in `task.json.coupling`. A repair stays in the same episode and obtains a fresh scored prediction. A forecast's valid status is separate from its execution permission. Inspect saved forecasts with `orx inspect --bank predictions --prediction P`; execution uses the selected original prediction without another model call.

## What the signals and memory mean

G estimates original-task acceptance given faithful execution of the proposed method. C predicts raw remaining resource use; common comparable dimensions use independent cost scales. R estimates one attempt's technical implementation/execution failure. U uses G/C/R and records missing terms. Binary H+ is frozen in that same prediction, stays outside U and schedules one post-task review only for an actually executed positive candidate. Failed positive attempts still need review; unexecuted candidates stay unobserved.

The Execution Evidence Bank retains real methods, attempts, checks, artifacts and costs in a bounded episode window. The Strategic Knowledge Bank retains reusable `when / do / expect` rules with immutable versions and evidence references. A published rule is the author's supported use decision; task acceptance, review completion and learning observations are separate facts. Existing cases with reliable derivation or small executable checks can support scoped cold-start publication and a substantive learning positive. A third transfer task, boundary test and regression suite are not a uniform prerequisite. Later checks report their own coverage.

Retrieval discovers material; it does not choose the solving method. Read `recommendations_basis`, vector hits and `degraded` separately. In the current implementation `--memory-mode none` clears structural recommendations but still runs vector retrieval; it is not a no-memory isolation switch. Missing measurements and unknown applicability remain explicit.

Actual adoption uses `execute --used-entry-ids ENTRY@VERSION`. Optional utility actions are separate: `keep` preserves status, `demote` suspends the selected version, and `retire` moves the entry to the cold archive. A `refuting` effect without `lifecycle_action` can also demote the adopted version in this checkout. See the reference before submitting an effect attribution.

## Runnable workflows

These use isolated memory homes, scripted predictions and real Python integer solves:

```bash
PYTHONPATH=src python3 references/examples/unified_solving.py --mode single
PYTHONPATH=src python3 references/examples/unified_solving.py --mode multi
PYTHONPATH=src python3 docs/validation/run_review_eight_demo.py
```

The first two exercise normal CLI planning, execution, checking, recording, close-out and H+ settlement. The A/B example illustrates an endpoint mechanism, explicit scoped publication, a later adopted version and a counterexample. The complete new/revised, covered and insufficient review commands have one canonical home: [references/induction.md](references/induction.md#complete-review-submissions).

## Reference map

| Need | Reference |
|---|---|
| Commands, result fields, configuration and compatibility/recovery paths | [commands.md](references/commands.md) |
| Self-check layers, failure handling and worked examples | [self_check.md](references/self_check.md) |
| Minimal input, modeling representation and CIR | [modeling.md](references/modeling.md) |
| U permission, G/C/R, resource scales, missing terms and budgets | [strategy_outcome.md](references/strategy_outcome.md) |
| Acceptance scope, terminal states, calibration and H+ dispatch | [episode_closeout.md](references/episode_closeout.md) |
| H+ review dispatch, submissions, learning receipts, and rule management | [induction.md](references/induction.md) |
| Prediction inputs, output fields and request audit | [world_model_contract.md](references/world_model_contract.md) |
| Input lifetime, historical records and memory maintenance | [snapshot_lifecycle.md](references/snapshot_lifecycle.md) |
| Design concepts | [concepts.md](references/concepts.md) |

## Development and experiments

```bash
PYTHONPATH=src python3 references/examples/_check_docs.py
PYTHONPATH=src:tests/harness python3 -m unittest discover -s tests/harness -q
```

See [H+ experiment modes](references/induction.md#learning-feedback-and-experiment-modes) and [fixed-memory testing](references/snapshot_lifecycle.md#fixed-memory-testing-with-per-question-copies) for the available experiment paths.

## License

MIT
