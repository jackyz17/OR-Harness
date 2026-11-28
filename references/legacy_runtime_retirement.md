# 历史运行机制退役审计

以下是精简提交 `44bd203` 的历史审计记录，规模统计和“未合并 main”等说明均指当时状态。后续 main 集成同时保留最新 Skill/reference 文档和历史观测关联修复；求解 Agent 继续使用 `plan-next → 编写代码 → execute → check/record/closeout → H+`，无需增加操作步骤。

本轮在 `refactor/retire-legacy-runtime` 删除 M2/M6 运行能力，保留当前 G/C/R/H+、实际事实记录和历史只读解析。相关回归 **422/422 通过**；固定输入下的有效预测请求、评分、选择、执行检查和归纳结果保持一致。

基线为 `origin/main` 的 `b356a89d779fbe3b28b0e50e2736f5abe17f98a5`，包含 G v2 提交 `6743b3e604999da504868e2f8894dc5f99027608`。修改位于独立 worktree `/tmp/or-harness-retire-legacy-runtime`。原工作区 main 保持 `c2a7b3d21e53cbeb294130f933b523cc155e39c7`，未修改实验部署、Runner、真实记忆库或数据集，未启动建库实验，未合并或推送 main。

机器可读的统计、完整限定符删除清单及文件摘要见 [审计数据](legacy_runtime_retirement.json)。该数据只用于本次审计，不参与框架运行。

## A. 删除清单

| 位置 | 删除内容及原调用方 | 删除依据及当前替代路径 |
|---|---|---|
| `world_model/service.py`（删除整个文件） | `PredictionService` 及其生成、绑定、比较、保存、请求组装和反馈计算；原由 `ORHarness.__init__` 和旧 API 调用 | 只有已退役 M2 API 使用该运行服务。正常预测只初始化 `StrategyOutcomeService`，使用 `wm-so/1` |
| `world_model/prediction.py`（删除整个文件） | `validate_prediction_payload`、`OutcomePrediction.new_id`；原由 M2 服务调用 | 新预测使用现有 `StrategyOutcomePrediction` 校验。仍有消费者的 `ActionSpec` 和历史 `OutcomePrediction` 解析/序列化移入已有 `contracts.py`，未增加兼容服务或新运行模块 |
| `api.py` | `predict_outcome`、`bind_outcome`、`compare_prediction`；原供旧服务调用及旧测试/示例使用 | 当前决策和执行原本已使用 `predict_strategy_outcome`、`bind_strategy_outcome`，本轮不改其行为 |
| `core/storage.py` | `put_world_model_prediction`；原仅由 M2 `_save` 调用 | 删除旧预测写入能力；原表和查询方法继续承担历史读取与已发生预测调用的预算记账 |
| `world_model/knowledge.py`（删除整个文件） | `KnowledgeTarget`、`learning_needs`、`knowledge_value`、`structural_need`、`precondition_realizability`、`validate_knowledge_changes` 及专用辅助函数；原由 M2 服务、上下文构建和 M6 API 调用 | 正式选择只使用 G/C/R，H+ 已有独立冻结门控和归纳作业，无 K 值运行消费者 |
| `world_model/knowledge_track.py`（删除整个文件） | 执行/归纳阶段知识变化评价、阶段 verdict、归因去重、可靠性统计和查表等函数；原由 `record`、`_induce_impl` 和 M6 API 调用 | 这些统计只评价旧 `knowledge_changes` 预测。当前 G/C/R 校准、H+ 回执和知识条目实际效果评价各有现存路径 |
| `api.py` | `knowledge_targets`、`prediction_reliability_table`、`evaluate_knowledge_execution`、`evaluate_knowledge_consolidation` | 删除 `build_prediction_context`、`record`、`_induce_impl` 中的旧调用及其 `knowledge_feedback`/错误返回挂钩；不改真实 `knowledge_delta`、索引同步、知识写入或 H+ 回执 |
| `api.py` 专用私有函数 | `_budget_override_view`、`_call_cost_dims`、`_charge_call_cost`、`_prediction_targets`、`_record_knowledge_verdict`、`_match_target`、`_induction_transition`、`_claim_change` | 原调用方均属于本轮退役链路；当前服务已有自己的预算检查、调用成本记录及实际知识写入事实 |
| `world_model/context.py`、`conditions.py` | `knowledge_targets_from_context`、`cell_evidence_from_stats`，旧目标/可靠性生成和目标版本写入，M6 可靠性对 capability 的补充 | 当前上下文不再生成这些块。候选相关的冻结知识引用直接保留到 `coverage.knowledge_layers`；真实 cell statistics、检索、能力证据及当前校准保持有效 |
| `provider.py` | M2/M4 两套旧 system prompt 和按旧 action 选模板的分支 | HTTP adapter 在网络调用之前拒绝非 `wm-so/1` 请求，保留原有 `unsupported prediction protocol` 错误前缀；当前 system prompt 和发送逻辑不变 |
| `api.py`、`planner.py`、`contracts.py` | `delta`、`prediction_mode`、`LEGACY_MODE_TO_KIND`、`prediction_kinds_for_mode` 及无效运行分支 | 审计确认它们已经不影响当前 U；不改 alpha/beta/gamma、共同 C/R 维度、缺项处理或稳定排序 |
| `cli.py` | 全局及 `plan-next` 的 `--delta`、`--prediction-mode`，共 4 处参数声明 | 删除无效开关；没有删除正式 CLI 命令 |
| `planner.py` | `PathStep`、`CandidatePath` 及其运行结构 | 新规划仍只接受 horizon=1，horizon=2 继续在预测前拒绝。旧 `paths` 保留为原始字典供读取，不再构造 rollout 类型或评分 |
| 文档与测试 | 旧 M2/M6 操作示例、导入路径和废弃功能专属测试 | 删除 `test_world_model_m2.py`、`test_world_model_m6.py`、`test_world_model_m6_closeout.py`；保留并迁移历史读取、上下文冻结、当前决策和归纳有效断言 |

本轮删除的历史公共运行入口共 **7 个**：M2 的生成/绑定/比较 3 个，M6 的目标/可靠性/执行评价/归纳评价 4 个。私有函数、存储 writer 和已删除模块内部方法不重复计入此数。

## B. 保留清单

| 保留内容 | 真实消费者与理由 |
|---|---|
| `ActionSpec` | 当前候选归一化、规划、CLI 和窗口契约仍使用；迁移到 `contracts.py` 并由 `world_model` 导出 |
| `OutcomePrediction`、`get_prediction`、`predictions_query`、`read_prediction` | 历史记录解析和 `inspect --bank predictions` 仍需要。API 直接读 Store，不初始化旧服务、不绑定、不评价、不重写历史记录 |
| 旧 `knowledge_changes`、context targets/reliability/cell_evidence 和 plan paths | 非空已存字段仍可读；原始数据库未迁移。新上下文不生成旧字段，旧预测不进入当前版本校准 |
| `PredictionServiceStatus` | 当前契约区分 provider 已配置、服务已实现、预测实际完成，属于有效状态 |
| capability evidence、冻结知识引用、知识采用和效果事实 | 当前预测、执行、归纳和生命周期仍有消费者；删除 M6 可靠性补充不等于删除 Harness 能力证据 |
| `predict_cost` / `predict-cost`、`PredictionSnapshot`、`compute_cost_feedback`、record 成本反馈、stats/selector 成本事实 | `src/or_harness/experiments/runner.py:310` 仍调用 `predict_cost` 并把结果交给 `record`。这一路描述单次 attempt，不冒充当前剩余任务 C；仓库外依赖也尚未排除，因此本轮保留 |
| `none/cases/strategic/cost-aware` memory modes | `strategy/selector.py` 和实验 Runner 中确有分支控制，不属于无效预测模式 |
| Runner 中的 `PREDICTION_MODES` 常量 | 引用审计显示仅剩未使用声明，没有实际实验控制作用；按“不修改实验组件”的边界原样保留。Harness 和 CLI 已无该开关 |
| H+ 作业、冻结模式、完成回执、compact submission、按需详细证据 | 当前正式链路仍在使用，未修改门控、取材标准或发布要求 |
| `hypothetical` / `source` 事实来源标记及过滤 | 用于旧数据审计，以及阻止想象记录混入真实进度、预算、执行统计和校准；不是可执行的多步规划服务 |

归纳路径审计结果：

```text
正式 H+ 取材：build_induction_material → require_job / material_for_job → build_compact_material
manual 维护：build_induction_material → start_decision → compact 或 detail
compact 提交：normalize_submission → 已有 relations/review 写入路径
实际知识写入：_submit_relations → 既有 rule_versions / 索引与生命周期处理
```

`induction_material.py` 仍直接使用 `maintenance.py` 的材料、既有知识和相关历史辅助函数；detail view 是有效 API，按需证据读取仍在使用。`induction_gate.py` 的 decision/meta 函数也有当前调用方。统一这些路径会涉及取材和维护行为，本轮只删除 M6 评价挂钩，未强行合并。

## C. 代码规模

统计对象为 Git 跟踪的 `src/` 源文件，修改后排除已删除文件；不含 `__pycache__`、`.pyc`、未跟踪生成文件、测试和文档。行数按 `splitlines()` 统计；`py.typed` 为空。修改前字节数已对 Git 基线重新核对。

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| src 总源码字节数 | 3,126,465 | 2,955,439 | -171,026 |
| Python 文件数 | 69 | 65 | -4 |
| 源码总行数 | 61,479 | 58,031 | -3,448 |
| api.py 行数 | 7,859 | 7,142 | -717 |
| CLI 命令数 | 30 | 30 | 0 |
| 本轮累计删除的历史公共运行入口数 | 0 | 7 | +7 |

没有新增运行模块、代理、适配层或迁移框架。

## D. 行为一致性与回归

修改前先运行代表性基线：159 项中 156 通过，2 失败、1 错误。失败均为已有断言滞后：plan-next 自动选择、决策/选择 action 数、已消耗 context 的读取。扩展相关覆盖后，又在未改动的基线 checkout 重现 5 处已有失败：2 处 M3 旧 C/选择断言，3 处 CLI 旧 capability 查询或旧 G/C fixture。修正测试使用当前接口和冻结输入引用，未恢复废弃机制，未降低有效检查标准。

末次静态清理删除了旧服务专用的未使用 import，以及无任何仓库调用方的 `CAPABILITY_EXECUTABLE_OPERATIONS` 常量；随后 14 项清理守卫再次通过。

本轮发现并修正了一个兼容细节：HTTP 拒绝旧协议时继续保留已有错误前缀，同时明确只支持 `wm-so/1`。

最终相关测试命令：

```bash
PYTHONPATH=src:tests/harness python -m unittest \
  tests.harness.test_cleanup_audit \
  tests.harness.test_command_surface \
  tests.harness.test_strategy_outcome.TestUnifiedSolving \
  tests.harness.test_prediction_context \
  tests.harness.test_g_observation_v2 \
  tests.harness.test_g_feedback_chain \
  tests.harness.test_c_signal_review \
  tests.harness.test_r_technical.TestTechnicalRLabels \
  tests.harness.test_r_technical.TestCommonMaxRScoring \
  tests.harness.test_episode_closeout.TestCloseoutBoundaries \
  tests.harness.test_wm_calibration_loop.TestNoDoubleCounting \
  tests.harness.test_hplus_binary \
  tests.harness.test_induction_submission \
  tests.harness.test_stage_e_history_retrieval.TestEndToEndHistory \
  tests.harness.test_stage_f_selfcheck_induction.TestReviewFlows \
  tests.harness.test_world_model_m3.TestPlanning \
  tests.harness.test_world_model_contract
```

结果：**422 项，83.603 秒，全部通过**。未运行全量测试套件。

| 覆盖范围 | 相关测试数 | 结果 |
|---|---:|---|
| 清理边界、CLI、历史读取、现行契约 | 14 + 79 + 82 | 通过 |
| 当前规划与候选选择 | 14 + 13 | 通过 |
| 冻结上下文与预测材料 | 102 | 通过 |
| G v2 观测及反馈 | 10 + 14 | 通过 |
| C 观测与历史材料 | 18 | 通过 |
| R 标签和共同事件评分 | 11 + 17 | 通过 |
| closeout 及校准去重 | 7 + 1 | 通过 |
| H+、归纳提交、取材与正式发布 | 14 + 19 + 2 + 5 | 通过 |

固定输入对照使用 [retirement_probe.py](../tests/harness/retirement_probe.py)：确定性时间/ID、本地 scripted provider、两个候选、一次临时执行、检查/记录/closeout 和 H+ review。修改前重复采集两次，再与修改后对照。探针会自行创建并清理临时记忆库，不读取实验库、不调用真实模型。

```bash
# 在基线 checkout 中运行该探针；它从当前工作目录解析被测源码与 fixtures。
PYTHONPATH=src:tests/harness python /tmp/or-harness-retire-legacy-runtime/tests/harness/retirement_probe.py /tmp/retire-legacy-before.json
# 在重构 worktree 中比较：
PYTHONPATH=src:tests/harness python tests/harness/retirement_probe.py /tmp/retire-legacy-after-final.json /tmp/retire-legacy-before.json
```

对照结果：每候选一次预测，共 2 次；有效请求逐字段一致；完整候选评分一致，仍选择 S04；执行检查 passed，G=1 且版本为 `wm-g-faithful/2`，R 观测一致；H+ 调度 1 个作业并返回相同 `insufficient_material` 回执。

允许的请求差异仅为：删除旧 delta/mode 后的配置摘要、把错误的 prompt 元数据标记 `wm2/2` 改为 `wm-so/1`、删除无消费者的 M6 空目标注释，以及对应非核心尺寸诊断。比较脚本仅归一化这些差异，保留任务、候选、证据、预算限制、输出契约和其余请求字段的严格比较。

正常流程中的旧调用由 `knowledge_targets=2`、`prediction_reliability_table=1`、`evaluate_knowledge_execution=1` 降为全部 0；本探针的空归纳写入分支前后均未调用 consolidation，真实写入路径另外由归纳回归覆盖。

此外，19 个受保护源码文件与基线字节一致，覆盖 G/C/R 观测、closeout、implementation facts、检索、请求预算、H+、归纳、selector/stats/rule_versions 及实验代码。当前 `strategy_prediction.py` 除模块说明外 AST 一致；system prompt SHA256 为 `2613510fe6499ce7774e2a5e0ef8a5e9a2f27d69dc4f58c971a5e25bbe9d6d5d`。U 评分及 benefit convention 函数 AST 一致。

文档检查 `_check_docs.py`、`prediction_context.py` 和 `contract_roundtrip.py` 示例通过，`git diff --check` 通过。完整测试日志、原始探针结果保存在本机 `/tmp/retire-legacy-*`，其关键结果和摘要写入审计 JSON；未向仓库加入大体积请求快照。

仓库外是否仍有调用已退役 Python API 的消费者，无法由本仓库证明。本轮保留已确认有实际消费者的成本接口；旧 M2/M6 API 的移除保持在独立分支。现行调用方可从 `or_harness.world_model.contracts` 或 `or_harness.world_model` 导入 `ActionSpec`，不再传递 `delta` / `prediction_mode`。

## E. 下一阶段建议

1. `api.py` 仍有 7,142 行，同时处理公共入口、上下文组装、执行绑定、自检与知识维护；`episode_closeout.py` 有 7,388 行，承担观测、版本兼容、发布和校准。后续可按真实职责审计边界，再考虑拆分，先保持事务顺序与公共行为。
2. `context.py` 仍有 2,925 行，包含证据投影、冻结兼容和请求组装。后续可先整理活跃输入与只读历史解析的职责，但要保留预算、裁剪和原始事实保障。
3. `maintenance.py` 的 1,539 行仍同时支持正式 H+ 和 manual/detail 维护。应先对 compact/detail 的证据集合、分页和提交语义建立等价性测试，再讨论统一取材；现有共享知识 writer 无需重建。
4. 若要退役独立 attempt 成本预测，应先确认 Runner/baseline 与仓库外消费者的用途，再决定如何保留成本事实；不能把单次 attempt 成本直接换名为剩余任务 C。实验文件中的未使用模式常量可由实验组件单独清理。

上述建议未在本轮实施。
