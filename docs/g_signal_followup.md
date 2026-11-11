# G 验收归因、反馈和统计修复

审查基线与起始远端 HEAD：`02771f87359ee3bd7710d185adade70610fdaa99`。本轮开始时已通过 GitHub 的 main 引用和完整树核对，服务器源码/测试/文档与该提交一致；修改前副本保存在 `/tmp/or-harness-g-fix-before`。工作区的 `.git` 仍为空的只读占位目录，远端提交通过 GitHub 连接完成。

本轮保持 `effective_completion/task_check_passed_given_faithful_execution`、`probability`、`wm-g-faithful/1`，不改 C/R、效用公式或候选入口。

| 问题 | 修复位置 | 基线复现 → 修复结果 |
|---|---|---|
| P1 弱验收 | `g_signal._acceptance_support/observe_faithful_g`；`verification.verify_task_result` 记录的检查及 `reference_source` | 自参照/未声明来源的 objective 检查仍 passed、independent=false，但 G 从错误的 1 改为 unknown。失败必须由真正运行且可信的失败项支持，不能借用早退后未执行的语义探针。充分语义验收仍产出 0/1；独立参考也可支持覆盖充分的目标验收。 |
| P1 对照可比性 | `g_signal.actual_solving_conditions/same_task_comparisons`；`episode_closeout._paired_feedback_row`；`strategy_prediction._save_with_conditions` | 候选均省略时限而实际回执为 15/180 秒时不再配对；同一 acceptance_id 下更换阈值、容差或 checker_version 也不再配对。同规则早退失败与通过仍可配对；两边都通过仍平局。 |
| P2 条件统计适用性 | `g_signal.conditional_g_statistics/select_conditional_g`；`context.provider_view`；`build_strategy_outcome_request` | scheduling 分组保留在审计里，不进入 assignment 请求。相同实际配置不再因预测累计多用 3 tokens 拆组，60/60.0 也归入同组。最多六个适用条目在体积裁剪前选择，已知结构/求解条件冲突被过滤，未知维度和稀疏支持明确标注。 |
| P2 观测与报数解耦 | `g_signal.g_observation_convention`；close-out 观测/评估及 g_observed_fact；`conditional_g_statistics` | provider 整个省略 benefit 时仍从框架冻结的请求契约观测真实执行：充分验收后通过数 1、概率评分数 0、Brier=null、分箱为空，预测仍为 not_predicted。显式旧指标继续用旧口径。 |

实际配置复用 `ExecutionRecord.solver` 和 `execution_features.execution_config.values/sources`，保留来源及执行引用。执行器设置与 action_id 匹配的脚本回执可提供观测；仅下发的配置指令不算实际回执。缺失/null 关键时限、来源不足、配置冲突或执行范围内条件不一致时，只保留案例，不建立可比对照。

`acceptance_rule` 的摘要/指纹从完整检查规范生成，覆盖检查器/参考版本、阈值、容差及其他检查参数，不包含结果值或归因自述。即使检查失败提前返回，规则身份仍完整；字典键顺序和等值数值拼写不改变身份。历史报告缺少规范时，不用一个名称或部分已执行检查补造规则身份。

两处 G 指令过滤继续有效：顶层 `prediction_reminders` 和 `strategy_outcome_calibration.reminders`。完整分组留在审计统计；provider 请求只见适用的有限子集。已有缓存可运行 `orx calibration --rebuild`，从已有记录重算发布，不调用 provider/solver，不改写原预测或原评价。

## 复现和测试

先在隔离副本中运行 02771f8 源码及新增复现测试：8 项测试全部失败（含子情形共 10 个失败），确认四类问题仍成立。实际 15/180 秒在执行记录的配置回执中另行断言，未用候选声明代替。

- 定向兼容性：202/202 通过（`/tmp/g-fix-directed-2.log`），覆盖世界模型、校准、task-check、旧快照等价性及发布性能。
- 最终 G 端到端专项：28/28 通过（`/tmp/g-fix-final-e2e.log`），含新增 12 项复现和既有归因/统计测试。
- 当前完整源码全量：`PYTHONPATH=src:tests/harness python -m unittest discover -s tests/harness`，**1742/1742 通过，128.014 秒**（`/tmp/g-fix-full-current.log`）。
- `python references/examples/_check_docs.py`、改动文档链接、`compileall` 与正常执行的新请求示例检查通过。

所有回归以替身 provider、临时执行工作区和隔离库运行；HTTP 测试只监听本机。

新增 12 个正常执行链回归位于 `tests/harness/test_wm_calibration_loop.py::TestGReviewReproductions`，复用 `g_fixtures.py`，走 predict → execute → check-task → close-episode → 下一次 provider 请求。覆盖全部复现，以及独立参考、未知实际时限、规则版本/容差、早退同规则、已知结构冲突、裁剪前筛选和完整审计保留。已有合法语义验收 0/1、同条件平局、旧口径、去重/重发布和双入口 C/R 兼容性继续验证。

[实际 provider 请求片段](g_signal_followup_request.json) 展示整个 benefit 缺失的案例：语义检查 passed，参考值来源 independent=false（没有独立参考值），合法语义验收仍支持 G=1；条件统计通过 1、概率评分 0。该数据来自隔离替身执行，用于确认链路接通，不是策略效果实验。

## 复跑命令

```bash
PYTHONPATH=src:tests/harness python -m unittest tests.harness.test_wm_calibration_loop.TestGReviewReproductions
PYTHONPATH=src:tests/harness python -m unittest discover -s tests/harness
python references/examples/_check_docs.py
PYTHONPATH=src python references/examples/g_signal.py
```

临时库、检查回执及合成任务是本轮全部验证资料。归因仍要求有来源的方法忠实度和原题覆盖审计；缺证据保持未知，不从 checker 名称或预测数值补标签。
