# 统一求解决策流程：实现与验收

本轮审查基线：`main@4e98fdf`。服务器目录没有 `.git` 元数据；已实际执行 `git status --short` 与 `git rev-parse --short HEAD`，两者均报告非 Git 仓库。因此通过 GitHub 远程 ref 核对 main，并逐文件比对初始快照：所有初始远程文件均与该基线一致。12 份用户本地文档保持原样，不纳入提交。

正常路径现为：`profile / recall → plan-next → execute --prediction → check-task → record → close-episode → 按冻结 H+ 接单`。本轮没有修改大规模冻结评测的隐藏 checker、复制、切分和统计协议。

## 实现与执行约束

| 位置 | 行为 |
|---|---|
| `api.py` 的 `plan_next` / `_plan_next_strategy_outcome` | 一个或多个候选共享冻结上下文，各调用一次预测，计算并保存 U、获选 P、逐候选执行许可、信号和缺项原因；稳定并列保留提案顺序。 |
| `planner.py` 的 `score_strategy_outcome_predictions` | `U=αG−βC−γR`。沿用 G 的可比口径、C 共同维度/组成去重和技术 R；缺项使用共同计分依据。没有可比 G 时退回；有 G 时可以得到部分 U。负 U 本身不新增禁行阈值。 |
| `api.py` 的 `_scored_decision` / `_resolve_execution_candidate` / `_claim_prediction_for_action` | CLI 与 Python 共用。启动前核对保存的 decision、P、有限 U、选中许可、任务/episode、方法、求解器、配置与消费状态。事务写入一次实际采用、预测 claim 和执行关联。兼容选择记录可复用，但不能授予落选候选权限。 |
| `world_model/prediction.py` 与 `contracts.py` | 经 ActionSpec 保留候选 config、方法前提、停止条件和 window；不再在评分入口丢失这些绑定依据。 |
| `world_model/hplus.py` | 新协议仅对真实评分采用且有执行结果的候选按事前 H+ 派单；拒绝、影子评分和未执行预测不产生结果或归纳待办。失败尝试仍消费原 P，修复需要新决策。 |
| `cli.py` | plan 摘要直接交出可执行 P/U 或退回原因；execute 引导检查和保存失败记录；重复 close 仍显示未完成的 `induction_schedule.pending_ids`。 |

独立 `predict-strategy` 用于查看预测。裸 strategy/solver、独立预测 ID、无有效 U、落选 P、手工选择、配置/任务冲突以及重复消费均不能授权正式执行。新预测的事后 bind 也必须对应原有评分采用。旧预测及历史绑定保留读取口径。

## C 的独立尺度

每维使用 `clip(log1p(predicted_d)/log1p(peak_d),0,1)`，peak 必须为正。优先读取显式 `PlanLimits.cost_scales` / CLI `--cost-scales`；否则从决策前发布且对所有参与候选条件适用的可信实测范围取最大值。该历史沿用执行宿主、计量窗口、阶段、结构和求解限制的适用性筛选。

预测值和本轮预测调用消耗都不用于构造 peak。相同维度相同尺度下，一个候选的归一值不受其他候选成本数值影响。缺可靠尺度的维度退出，显式预测零保留；共同维度确定、组成去重后，沿用原维度权重并在实际集合上归一。保存原始资源值、peak、来源、窗口、实际权重与缺项原因。`solver_runtime_s`、单次沙箱 `latency_s` 与端到端 `remaining_latency_s` 沿用各自口径。

示例的 `tool_calls peak=30` 是显式的替身配置，不是框架默认值或通用资源标尺。另有正常执行/记录/结算回归证明，下一题可从先前实测范围取得 peak=30，并保留事实来源。

## 两条可复现命令

从仓库根目录执行，使用本机临时替身 provider 和两个全新隔离库；脚本实际调用 CLI，不使用测试授权 helper。

```bash
PYTHONPATH=src python3 references/examples/unified_solving.py --mode single
PYTHONPATH=src python3 references/examples/unified_solving.py --mode multi
```

每条命令都生成原题、候选和脚本，走完整评分、执行、检查、记录、收尾及归纳链。内部 `plan-next` 的两个实测命令（本机 provider 端口仅在该次脚本运行中有效）：

```bash
/data/home/zhanyiqi/miniforge3/bin/python3 -m or_harness.cli --home /tmp/orx-unified-ah3b7e_r --world-model http://127.0.0.1:44913::scripted-example plan-next --task /tmp/orx-unified-ah3b7e_r/ws/endpoint-source.json --episode E --candidates /tmp/orx-unified-ah3b7e_r/ws/endpoint-source-candidates.json --cost-scales '{"tool_calls":30}'
/data/home/zhanyiqi/miniforge3/bin/python3 -m or_harness.cli --home /tmp/orx-unified-joe9065e --world-model http://127.0.0.1:36467::scripted-example plan-next --task /tmp/orx-unified-joe9065e/ws/endpoint-source.json --episode E --candidates /tmp/orx-unified-joe9065e/ws/endpoint-source-candidates.json --cost-scales '{"tool_calls":30}'
```

| 实测 | 单候选 | 多候选 |
|---|---:|---:|
| 每轮候选数 / 模型调用数 | 1 / 1 | 2 / 2 |
| 两题累计模型调用数 | 2 | 4 |
| endpoint 的原始 tool_calls 预测 | 3 | 3 |
| endpoint 的 C / U | 0.403698 / 0.396302 | 0.403698 / 0.396302 |
| 另一候选的原始 tool_calls / U | 不适用 | 30 / −0.2 |
| 第一题冻结 H+ / 待办 | 1 / 一份 | 1 / 一份 |
| 第二题冻结 H+ / 新待办 | 0 / 无 | 0 / 无 |

G=0.9、技术 R=0.1；同一候选同维度使用同一尺度的 C 和 U 保持一致。H+ 不进入 U。execute 复用原预测，没有额外模型调用；每个真实尝试只记一次实际采用。

原题最小化 `2X−5Y`，非负整数且 `X+Y≤641`、`X≥2Y+130`。实测 X=470、Y=170、目标 90；脚本与完整枚举比较，checker 核对独立参考目标、整数、重算目标及两条原题不等式。第一题派发一份正信号 job，取材后同时提交 relation/review；回执关联原 P、真实归纳动作、源 EX 与版本 1，冻结核验记录为 checked，学习标签为 1。学习增量标签与原假设是否兑现分别记录。第二题使用实际版本并按 H+=0 跳过新归纳。重关不重复派单，原有条目可继续维护。

端点/非单调边界的补充实测：A=2699、B=90、边界=0，在第三道真实 C 题前基于推导与边界证据作有范围的发布，C=196。提交经 CLI 完整重放返回同一回执，没有新增归纳动作。

完整的真实请求片段、逐候选原始信号、评分依据、P/EX/J/归纳动作/版本和实际 CLI 命令保存在 [unified_solving_demo.json](unified_solving_demo.json)。数字来自替身预测，仅验收机制；没有据此声称真实模型排序或工业成本判别力得到提升。

## 回归结果

| 检查 | 实际结果 |
|---|---|
| `PYTHONPATH=src:tests/harness python3 -m unittest discover -s tests/harness -q` | 1,908 项通过，209.785 秒 |
| 定向 G/C/R、评分、CLI、H+、成本观测与快照回归 | 280 项通过，55.107 秒 |
| `test_hplus_experiments test_experiments` | 6 项通过，37.628 秒 |
| 原有执行前重复消费/工作目录拒绝等检查 | 4 项通过 |
| `PYTHONPATH=src python3 references/examples/_check_docs.py` | 通过 |
| 文档承诺的 14 个可运行示例 | 全部通过；增强后的单/多候选示例另各跑一次 |
| `python3 -m compileall -q src tests/harness references/examples` | 通过 |

新增回归覆盖单候选 U、共享上下文、同维度尺度稳定、历史尺度、共同缺项/显式零、负 U、H+ 不入 U、启动前拒绝、两连接争用一次采用、失败正信号空审阅以及完整 CLI 两种候选数。旧校准/历史窗口测试使用显式历史导入 fixture 维持原有样本语义；正式入口没有测试专用放行开关。

## 单列的归纳文档审核稿

[unified_induction_copy.md](unified_induction_copy.md) 单列 `SKILL.md` 与 `references/{induction,hplus}.md` 的对应拟用文案，含三种完整写法：

- 有据的新增/修订：同时提交 `--relation` 和 `--review`，区分发布范围与学习结论。
- 已覆盖/无增量：充分、非实质对照与真实事实引用支持学习标签 0，可空审阅或补已有版本证据。
- 材料不足：`insufficient_material` 完成空审阅，学习标签未知。

文稿末尾给出本轮真实 J/EX、原始 relation/review 的完整 CLI 重放命令。事前 H+=0 表示跳过归纳；它与真实审阅后的学习标签 0 不同。

## 完整改动文件

### 代码

- [src/or_harness/api.py](../../src/or_harness/api.py)
- [src/or_harness/cli.py](../../src/or_harness/cli.py)
- [src/or_harness/experiments/hplus.py](../../src/or_harness/experiments/hplus.py)
- [src/or_harness/experiments/runner.py](../../src/or_harness/experiments/runner.py)
- [src/or_harness/world_model/contracts.py](../../src/or_harness/world_model/contracts.py)
- [src/or_harness/world_model/cost_signal.py](../../src/or_harness/world_model/cost_signal.py)
- [src/or_harness/world_model/hplus.py](../../src/or_harness/world_model/hplus.py)
- [src/or_harness/world_model/planner.py](../../src/or_harness/world_model/planner.py)
- [src/or_harness/world_model/prediction.py](../../src/or_harness/world_model/prediction.py)

### 测试与 fixture

- [tests/harness/c_fixtures.py](../../tests/harness/c_fixtures.py)
- [tests/harness/decision_fixtures.py](../../tests/harness/decision_fixtures.py)
- [tests/harness/g_fixtures.py](../../tests/harness/g_fixtures.py)
- [tests/harness/review_fixtures.py](../../tests/harness/review_fixtures.py)
- [tests/harness/test_budget_stop.py](../../tests/harness/test_budget_stop.py)
- [tests/harness/test_c_signal_review.py](../../tests/harness/test_c_signal_review.py)
- [tests/harness/test_calibration_v2.py](../../tests/harness/test_calibration_v2.py)
- [tests/harness/test_capability_evolution.py](../../tests/harness/test_capability_evolution.py)
- [tests/harness/test_capability_trace.py](../../tests/harness/test_capability_trace.py)
- [tests/harness/test_command_surface.py](../../tests/harness/test_command_surface.py)
- [tests/harness/test_cost_eligibility.py](../../tests/harness/test_cost_eligibility.py)
- [tests/harness/test_documented_result_keys.py](../../tests/harness/test_documented_result_keys.py)
- [tests/harness/test_end_to_end_cli.py](../../tests/harness/test_end_to_end_cli.py)
- [tests/harness/test_episode_closeout.py](../../tests/harness/test_episode_closeout.py)
- [tests/harness/test_evidence_window.py](../../tests/harness/test_evidence_window.py)
- [tests/harness/test_harness_feedback.py](../../tests/harness/test_harness_feedback.py)
- [tests/harness/test_hplus.py](../../tests/harness/test_hplus.py)
- [tests/harness/test_hplus_binary.py](../../tests/harness/test_hplus_binary.py)
- [tests/harness/test_linkage_attribution.py](../../tests/harness/test_linkage_attribution.py)
- [tests/harness/test_method_evidence.py](../../tests/harness/test_method_evidence.py)
- [tests/harness/test_no_catalog.py](../../tests/harness/test_no_catalog.py)
- [tests/harness/test_numbered_entries.py](../../tests/harness/test_numbered_entries.py)
- [tests/harness/test_prediction_context.py](../../tests/harness/test_prediction_context.py)
- [tests/harness/test_r10_end_to_end.py](../../tests/harness/test_r10_end_to_end.py)
- [tests/harness/test_r12_world_model.py](../../tests/harness/test_r12_world_model.py)
- [tests/harness/test_r13_remaining_cost.py](../../tests/harness/test_r13_remaining_cost.py)
- [tests/harness/test_r14_induction.py](../../tests/harness/test_r14_induction.py)
- [tests/harness/test_r15_recall_pairing.py](../../tests/harness/test_r15_recall_pairing.py)
- [tests/harness/test_r19_solver_duplicate.py](../../tests/harness/test_r19_solver_duplicate.py)
- [tests/harness/test_r22_publish_performance.py](../../tests/harness/test_r22_publish_performance.py)
- [tests/harness/test_r23_budget_labels.py](../../tests/harness/test_r23_budget_labels.py)
- [tests/harness/test_r24_archive_scan.py](../../tests/harness/test_r24_archive_scan.py)
- [tests/harness/test_r7_semantics.py](../../tests/harness/test_r7_semantics.py)
- [tests/harness/test_r_technical.py](../../tests/harness/test_r_technical.py)
- [tests/harness/test_review_eight.py](../../tests/harness/test_review_eight.py)
- [tests/harness/test_rule_versions.py](../../tests/harness/test_rule_versions.py)
- [tests/harness/test_snapshot_lifecycle.py](../../tests/harness/test_snapshot_lifecycle.py)
- [tests/harness/test_strategy_outcome.py](../../tests/harness/test_strategy_outcome.py)
- [tests/harness/test_task_check.py](../../tests/harness/test_task_check.py)
- [tests/harness/test_usage_accounting.py](../../tests/harness/test_usage_accounting.py)
- [tests/harness/test_vector_recall.py](../../tests/harness/test_vector_recall.py)
- [tests/harness/test_verification_honesty.py](../../tests/harness/test_verification_honesty.py)
- [tests/harness/test_wm_calibration_loop.py](../../tests/harness/test_wm_calibration_loop.py)
- [tests/harness/test_world_model.py](../../tests/harness/test_world_model.py)
- [tests/harness/test_world_model_m2.py](../../tests/harness/test_world_model_m2.py)
- [tests/harness/test_world_model_m3.py](../../tests/harness/test_world_model_m3.py)
- [tests/harness/test_world_model_m6.py](../../tests/harness/test_world_model_m6.py)
- [tests/harness/test_world_model_m6_closeout.py](../../tests/harness/test_world_model_m6_closeout.py)

### agent 文档、示例与验收材料

- [README.md](../../README.md)
- [README_zh.md](../../README_zh.md)
- [SKILL.md](../../SKILL.md)
- [docs/validation/run_review_eight_demo.py](../../docs/validation/run_review_eight_demo.py)
- [docs/validation/unified_induction_copy.md](../../docs/validation/unified_induction_copy.md)
- [docs/validation/unified_solving.md](../../docs/validation/unified_solving.md)
- [docs/validation/unified_solving_demo.json](../../docs/validation/unified_solving_demo.json)
- [references/commands.md](../../references/commands.md)
- [references/examples/_check_docs.py](../../references/examples/_check_docs.py)
- [references/examples/_result_keys.py](../../references/examples/_result_keys.py)
- [references/examples/_workflow.py](../../references/examples/_workflow.py)
- [references/examples/capability_evolution.py](../../references/examples/capability_evolution.py)
- [references/examples/capability_followup.py](../../references/examples/capability_followup.py)
- [references/examples/capability_gain.py](../../references/examples/capability_gain.py)
- [references/examples/end_to_end.py](../../references/examples/end_to_end.py)
- [references/examples/episode_closeout.py](../../references/examples/episode_closeout.py)
- [references/examples/g_signal.py](../../references/examples/g_signal.py)
- [references/examples/linkage_attribution.py](../../references/examples/linkage_attribution.py)
- [references/examples/method_evidence.py](../../references/examples/method_evidence.py)
- [references/examples/no_catalog.py](../../references/examples/no_catalog.py)
- [references/examples/prediction_context.py](../../references/examples/prediction_context.py)
- [references/examples/strategy_outcome.py](../../references/examples/strategy_outcome.py)
- [references/examples/task_check.py](../../references/examples/task_check.py)
- [references/examples/unified_solving.py](../../references/examples/unified_solving.py)
- [references/examples/usage_accounting.py](../../references/examples/usage_accounting.py)
- [references/examples/versioned_induction.py](../../references/examples/versioned_induction.py)
- [references/hplus.md](../../references/hplus.md)
- [references/induction.md](../../references/induction.md)
- [references/strategy_outcome.md](../../references/strategy_outcome.md)

