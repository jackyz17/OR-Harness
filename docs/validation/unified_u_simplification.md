# 统一 U 补充：入口精简、旧计算退役与 H+ 去重

工作起点为上轮 `6d184b726ae9a6b37d31a5f19086d5dc93648247`。目录没有 `.git` 元数据，已尝试 `git status --short`、`git rev-parse --short HEAD` 和 `git diff`；改用初始文件快照、远程 ref 及文件 SHA 比对。本轮涉及文件在开始时与上轮提交一致，原有 12 份本地文档保持不动。用于定位旧实现的 `4e98fdf` 缓存已逐文件核验 SHA；没有回退上轮统一 U、固定 C 尺度或执行绑定的有效改动。

## 决策入口

- `cli.cmd_plan_next` 只读取候选 JSON 列表，交给公共 `plan_next`。移除了 CLI 参数对象和 harness 实例上的临时拒绝列表。
- `_candidate_specs` 一次规整原始输入，返回有效候选与逐条拒绝原因，按有效候选数应用原有限制。身份冲突、非法候选不占有效名额。候选数量、调用数、时间和真实预算仍沿用现有规则。
- 规划给预测函数传入标准 `CandidateRef`，复用已有类型入口，避免预测时再次规整；原规整说明仍保存在预测诊断中。实测 6 条原始输入（4 好、2 坏）：CLI/API 均规整 6 次、预测 3 个有效候选，并返回相同拒绝原因。
- 一组候选评分行同时用于 CLI/Python 返回、`plan_next_evaluation` 持久化和执行许可检查。预算超限与正常返回共用结果组装，获准 P 与选中分数来自该组评分，不新增审批表或第二套状态。
- 无用量预测的 `decision_action_id` 在上轮已提前保存，本轮复用并去掉重复赋值。回归确认 token 和延迟均未报告时，G/C/R 有效预测仍归属原决策、拥有有限 U、可执行且只调用一次模型；费用保持未知，不补零。实际采用继续由原事务记录，复用原 P。
- `cmd_profile` 从已经得到的 `profile` 生成派生报告。Python `derivation_report(task, cir=...)` 保持兼容；CLI 字段不变，规整与派生回归确认只计算一次画像。

正常流程继续是 `plan-next → execute --prediction P → check-task → record → close-episode`。既有无 U/落选/重复消费/任务或配置冲突的执行前拒绝及未执行事实边界保留。

## 已退役的计算与保留部分

源码、包导出和仓库内 Python 脚本审查没有找到下列旧规划辅助函数的真实调用；只有本次同步退役的专用测试。已移除：

- `predicted_cost_vector`
- `terminal_quality`
- `terminal_risk`
- `evaluate_path`
- `comparison_norms`
- `real_incumbent_available`
- `build_hypothetical_successor`
- `second_step_dependency_ok`
- `_is_unusable_incumbent`
- `_looks_like_solved_answer`

同时移除其专用常量和不再使用的导入，并更新 planner 总览及直接相关注释。

继续保留：

- `PlanLimits`、`PathStep`、`CandidatePath`、`PlanResult` 的历史序列化字段及读取；新增往返回归保留 horizon=2、两个步骤、假想快照 ID、旧 Q/C/R 分解与知识项明细。
- `prediction.py`、`service.py`、`context.py`、`knowledge.py`、`knowledge_track.py` 及 API 研究预测/真实结算仍使用的知识契约、需要、预条件和证据追踪。研究模式的这些能力没有改口径。
- `knowledge_value` / `precondition_realizability` 等研究回归继续保留；没有把本轮规划死代码清理扩展成研究能力清理。

本轮删除的 10 个辅助函数没有需要保留的真实源码/脚本调用。旧计划读取不执行已退役的路径评分，也不会据此授权正式执行。

## H+ 读取和存储

`feedback` 先按 job 中的 `signal.family` 执行原有类别过滤，再读取回执、判定支持证据。cutoff、匹配排序、预算和标签口径不变。现有相近案例回归验证：异类回执观测调用数为 0，同类案例仍得到原来的回执及学习标签。

`complete` 仍把事实保存到原 `hplus_receipt`。新 `hplus_result` 只保留必要结果和 `receipt_ref`；首次返回保留完整 `induction_receipt`。`completed_result` 在读取时展开引用，同时接受旧的内嵌正文格式。API `induce` 的普通完成重试及事务内完成分支均使用该读取入口。检查过所有 `result_ref` 读取点，没有遗留直接假设新结果内嵌回执的生产读取。

回归覆盖首次完成、重开后相同提交重试、旧内嵌格式读取、延迟索引修复、下一轮预测反馈，继续关联同一 job、求解动作、真实归纳动作、回执和规则版本。H+=1 必须完成审阅、H+=0 跳过新归纳及学习标签 1/0/未知均沿用原义。

还用服务器上轮真实 `6d184b7` 验收库重放了原 CLI 提交，而非重建旧库：

- job：`ij_687aaca0be5371420445828f`。
- `already_completed=true`，仍返回原回执 `hplus_receipt:dafc1d15a125c1e73ff2b38d86b484094a165e2664ee700da0e6890f6c631038`。
- 原 `result_ref`、旧内嵌结果正文均未变化；归纳动作数仍为 1。

没有为这次去重迁移或重写历史库。

## 测试精简

- `test_world_model_m6.py` 退役 5 条旧计算/后继专用测试；`test_world_model_m6_closeout.py` 退役 1 条旧评分测试。
- `test_r17_hplus_followup.py` 的 6 条测试全部为文案断言，文件退役；`test_r16_assertion_symmetry.py` 移除 5 条帮助/文案断言，保留断言语义、claim 与 assertion 分离和真实归纳形状检查。
- 两个 `test_suggestion_is_not_a_selection` 合并到既有正常规划—实际采用回归中，覆盖计划记录、一次预测、无用量时决策 ID、实际采用，以及持久评分与返回评分一致。
- 已有候选拒绝测试扩展为 CLI/API 混合候选限流一致性；已有 H+ 重试、反馈、画像回归分别补充新旧格式读取、先筛选后读取及单次画像计算。
- `test_r_technical.py` 的独立尺度、同口径、缺项和显式零回归在上轮已经处理，本轮核对并复用；不再改回候选相对归一化。
- 复用 `_check_docs.py`：链接检查改用统一文档列表，覆盖 `references/hplus.md`，并检查归纳状态命令及 `--induction-id`。命令、参数、链接和可运行示例的机械检查继续使用原机制。

总体从 1,908 项到 1,890 项，净减 18：删除/合并 19 条旧或重复测试，替换一条候选测试（数量不变），补一条历史计划读取测试。

## 实际验证

| 命令/范围 | 结果 |
|---|---|
| `PYTHONPATH=src:tests/harness python3 -m unittest discover -s tests/harness -q` | 1,890 项通过，209.331 秒 |
| 实际触及模块及既有 C/R、H+、索引修复定向回归 | 385 项通过，52.756 秒 |
| 最后一次候选规整补充回归 | 38 项通过，2.609 秒 |
| `PYTHONPATH=src python3 references/examples/_check_docs.py` | 通过，含 H+ 文档链接及归纳参数 |
| `python3 -m compileall -q src tests/harness references/examples` | 通过 |
| 上轮真实旧库的原 CLI 提交重放 | 同一完整回执，无新动作，无历史结果改写 |

定向命令：

```bash
PYTHONPATH=src:tests/harness python3 -m unittest -q \
  test_candidate_normalization test_world_model_m3 test_strategy_outcome \
  test_command_surface test_harness_feedback test_hplus_binary test_review_eight \
  test_world_model_m6 test_world_model_m6_closeout test_r16_assertion_symmetry test_r_technical
```

没有改动 `SKILL.md` 与 `references/{induction,hplus}.md` 的 `--relation` / `--review` 示例正文。上轮单列的 [归纳文档审核稿](unified_induction_copy.md) 保持原样，本轮没有新增待审核正文。大规模冻结评测协议未调整。

## 改动文件

- [references/examples/_check_docs.py](../../references/examples/_check_docs.py)
- [src/or_harness/api.py](../../src/or_harness/api.py)
- [src/or_harness/cli.py](../../src/or_harness/cli.py)
- [src/or_harness/world_model/hplus.py](../../src/or_harness/world_model/hplus.py)
- [src/or_harness/world_model/knowledge.py](../../src/or_harness/world_model/knowledge.py)
- [src/or_harness/world_model/planner.py](../../src/or_harness/world_model/planner.py)
- [tests/harness/test_candidate_normalization.py](../../tests/harness/test_candidate_normalization.py)
- [tests/harness/test_harness_feedback.py](../../tests/harness/test_harness_feedback.py)
- [tests/harness/test_hplus_binary.py](../../tests/harness/test_hplus_binary.py)
- [tests/harness/test_r16_assertion_symmetry.py](../../tests/harness/test_r16_assertion_symmetry.py)
- `tests/harness/test_r17_hplus_followup.py`（删除：仅文案断言）。
- [tests/harness/test_strategy_outcome.py](../../tests/harness/test_strategy_outcome.py)
- [tests/harness/test_world_model_m3.py](../../tests/harness/test_world_model_m3.py)
- [tests/harness/test_world_model_m6.py](../../tests/harness/test_world_model_m6.py)
- [tests/harness/test_world_model_m6_closeout.py](../../tests/harness/test_world_model_m6_closeout.py)
- `docs/validation/unified_u_simplification.md`（删除：仅文案断言）。
