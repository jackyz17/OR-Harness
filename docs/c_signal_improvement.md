# C 信号判别力链路验收

审查基线：`main@10368fe9d72521730c8c64758422a29b2e5e23bc`。远端 HEAD 与基线一致；修改前已保存本地快照。源码、测试、运行时提示词和相关参考文档在本服务器修改、验证后提交 main；保留原有本地文件和未跟踪文档。

## 预测输入

`cost_signal.py` 从已发布的关闭 episode 中构造条件实测范围和案例。`context.provider_view` 在体积裁剪前，按执行宿主及其实际报告的模型、计量窗口、已知阶段、题族、已知结构和求解限制选择最多 4 个范围、3 个案例。执行宿主身份独立于世界模型 provider。首次建模需要明确的未建模状态；已有模型需要产物/可用性记录；缺失阶段或宿主模型身份时不给通用成本先验。

组内保留实际执行配置，缺省候选参数不会抹去已知时限；累计预测 token 消耗不作为分组条件。每维有 P50/P75、范围、样本数、独立 episode 数和来源。稀疏证据只支持已执行范围；缺失的结构、设置不被当作确认匹配。原始完整分组和覆盖率保存在审计发布块 `conditional_cost` 中，provider 收到选取后的 `cost_evidence`。

同一次世界模型调用先比较候选相对参考流程新增/省去的工作，再估计原始工具调用、tokens、时间等成本。历史总量中的步骤只计一次；依据写在 `evidence_basis`，无法估计的维度写在 `unsupported_fields`。`cost` 仍只允许原有资源维度的数值，`expected`/`expected_measured` 的含义保持一致。宿主轮次作为解释事实，不增加成本维度。

[请求片段](c_signal_request_example.json) 摘录了正常执行/结算后实际发给替身 provider 的两个请求：同一道供应链选厂题，A 沿用已有联合模型，改约束、求解一次并验收；B 重建模型、求解三个子问题、合并后验收。两份请求含各自步骤、可复用产物及两个已记录案例。测试宿主计量分别为 8、20 次工具调用，同条件范围 P50=14、P75=17、n=2；案例保留方法、实际资源、执行配置和来源引用。这些计量是隔离 fixture 的构造数据，只证明证据链路，不代表真实模型预测效果。

## 观测和统计

C 观测规则为 `wm-c-window/1`。默认窗口从共享决策锚点到任务真实终止，覆盖建模、求解、验收、返工和重试。

| 证据 | 处理 |
| --- | --- |
| 同窗口、可信来源的宿主计数器起止值 | 使用非负增量；tokens 为输入加输出 |
| 完整、带时间及调用 ID 的调用明细 | 按窗口裁剪、按 ID 去重；跨边界调用无法拆分时保持未知 |
| 重复引用同一完整窗口报告 | 同一事实，不重复累加；冲突总量/宿主保持未知 |
| 成功、最终失败、确认中止的 `finish_task` 或可信完整跨度 | 可提供真实终止边界和完整已花费资源 |
| 只有最后执行/验收结束时间 | 端到端时间仅为下界 |
| 只有边界不清的 attempt 总量 | 保留原口径和数值，不充当剩余窗口总量 |
| task 等其他计量范围的旧总量 | 保留记录，排除与 attempt/剩余窗口的误比较 |
| 后续离线执行或结算后新增动作 | 不进入原终止/close-out 边界内的成本窗口 |
| 脚本墙钟代理 `wall_proxy` | 保留原事实，不冒充精确内层求解时间 |

`record`、`report_action` 和已有 `amend-cost --usage-file` 保留可选窗口证据，完整宿主总量不覆盖 attempt 成本，也不再次叠加辅助动作成本。`close_episode` 只发布事实，不制造终止时间。已有证据可以提供宿主轮次、执行次数、报告过求解时间的 attempt、显式求解调用和已记录修复；不把每个执行 attempt 猜成一个内层求解调用。

`evaluations[].cost.observed_fact` 与模型是否报 C 无关。覆盖率分别报告 `n_actual`、`n_predicted`、`n_error` 和 `n_log_ratio`。实际窗口按执行/窗口身份去重；预测误差仍按各自冻结预测计算。省略整个 cost 的合格实测进入基线，有实际值 1、有预测值 0、误差样本 0。

每维保留绝对误差和带符号误差 `actual-predicted`，显式零参与统计。`solver_runtime_s` 与 `remaining_latency_s` 在各自口径可比时分别评分。log-ratio 只使用预测、实测都为正的样本，支持数取这一批样本自己的独立 episode 数。全局 log-ratio 留作审计；不生成全局或单例 C 调高/调低、乘数指令。两个 provider 提醒入口均过滤旧 G/C 指令，包括缓存上下文；R 和区间反馈继续保留。

效用归一化、阈值、缺项扣分、gate 及候选入口沿用现有实现。

## 回归验证

使用隔离临时库和替身 provider，经正常 `predict/plan → execute → record/check-task → finish_task → close-episode → 下一次请求` 路径覆盖上述机制。额外覆盖：共享锚点、完整调用明细的窗口裁剪、零值和 log 支持数、两种耗时、省略 cost、迟到报告重发布、CLI 回填、未知阶段、不同宿主/题族/时限、累计多 3 个预测 token 不拆组、重复结算不加样本和原始预测保持冻结。

```bash
PYTHONPATH=src:tests/harness python -m unittest \
  tests.harness.test_r13_remaining_cost tests.harness.test_cost_eligibility \
  tests.harness.test_calibration_v2 tests.harness.test_wm_calibration_loop \
  tests.harness.test_usage_accounting tests.harness.test_episode_closeout \
  tests.harness.test_r19_solver_duplicate
PYTHONPATH=src:tests/harness python -m unittest discover -s tests/harness
```

最终结果：定向 **272/272** 通过（34.369 秒），`tests/harness` 全量 **1765/1765** 通过（143.742 秒）。相对基线新增 23 项测试。请求 JSON 的两个提醒入口已核验无 G/C 调分指令。

## 真实判别力覆盖

只读检查本地 `or_harness_home/or_harness.db`：executions=0、contract_predictions=0、closed episodes=0；已有 1 条动作记录不提供真实策略配对。未修改该库。当前可比较的真实同初始状态策略对为 0。

| 对照组（优先 tool_calls） | 可比真实策略对 | 同题排序正确率 | 成本差值误差 |
| --- | ---: | --- | --- |
| 旧预测 | 0 | 未测 | 未测 |
| 同条件历史基线 | 0 | 未测 | 未测 |
| 历史基线加候选步骤 | 0 | 未测 | 未测 |

后续实验需冻结相同原题、初始模型/产物和状态，记录实际宿主模型及求解限制，让两种方法都真实运行到终止，并取得相同窗口的完整工具调用计量。历史基线只能使用决策前可见的关闭 episode，按 episode 划分评估数据，避免泄漏。报告可比策略对数、原始 `tool_calls` 排序（含平局）正确率及预测差值相对实际差值的绝对误差；tokens/时间在证据完整时分别报告。替身请求已验证接通；策略成本排序的改善仍待真实对照验证。

## 改动文件

| 范围 | 文件 |
| --- | --- |
| API/记录入口 | `src/or_harness/api.py`、`src/or_harness/cli.py` |
| C 证据及运行时预测 | `src/or_harness/world_model/cost_signal.py`（新增）、`context.py`、`strategy_prediction.py` |
| 计量、结算和反馈 | `src/or_harness/world_model/usage.py`、`episode_closeout.py`、`conditions.py` |
| Fixture 和回归 | `tests/harness/c_fixtures.py`（新增）、`g_fixtures.py`、`test_r13_remaining_cost.py`、`test_cost_eligibility.py`、`test_calibration_v2.py`、`test_wm_calibration_loop.py`、`test_episode_closeout.py` |
| 运行时及参考文档 | `SKILL.md`、`references/strategy_outcome.md`、`world_model_contract.md`、`prediction_context.md`、`episode_closeout.md`、`commands.md` |
| 验收产物 | 本文、`docs/c_signal_request_example.json` |
