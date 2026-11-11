# G 信号改造验证记录

日期：2026-11-11。审计基线：[main@5b4cec6](https://github.com/jackyz17/OR-Harness/commit/5b4cec6356c9428108dbfa4df1812bb44b423d92)。

当前挂载目录的 `.git` 是空的只读占位目录，不能读取本地 status/HEAD。修改前已保存 `/tmp/or-harness-g-before` 源码/测试/参考文档快照及散列清单；16 个指定核心、文档和测试文件的 Git blob SHA 与远端基线逐一匹配。按现有任务验收、方法记录、执行绑定、条件快照和 close-episode 发布实现扩展，没有覆盖其他工作区文件。

## 契约和归因

新请求固定 `effective_completion/task_check_passed_given_faithful_execution`、单位 `probability`、独立观测版本 `wm-g-faithful/1` 和候选声明的执行范围。G 是当前原题和约定求解条件下，忠实实现候选步骤后通过原题验收的概率。方法遗漏的约束仍然遗漏；脚本能否实现属于 R。G 必须是有限 JSON 数字且在 [0,1]，bool/数值字符串也不能充当 G。口径/范围/数值不合格时只使 G 不可用并保存原因，合格 C/R 和一句 `benefit.notes` 推演仍可读取。现有效用消费者识别并校验同一指标。

| 可核验事实 | 新 G 标签 |
|---|---|
| 实际忠实执行候选，检查覆盖所声明的原题验收且通过 | 1 |
| 实际忠实执行候选，核验确认方法本身导致原题不达标 | 0 |
| 实现漏写约束、改用其他方法、崩溃/超时无可评结果、检查不足或原因不清 | 未知，保留原因 |

可选 `check-task.g_attribution` 记录 `method_fidelity`、`failure_source`、`acceptance_scope`、版本化 `acceptance_id`、`source`、`assessment` 和代码/结果 `evidence_refs`。框架把引用解析到本次执行的产物散列，并核对绑定、实际方法、声明方法和执行范围。它是带来源的审计结论，不是从步骤文字相同或 checker 单独 failed 自动推定忠实度。完整字段见 [episode_closeout.md](../references/episode_closeout.md#g-attribution-evidence)。

旧预测/显式旧指标按原口径读取和计算误差；solver gap 仍是求解器自身的 `max(0,1-gap)`。旧单次无结果=0 的规则保留在旧 completion 适配器。历史执行若有充分新归因证据，可独立形成 `g_observed_fact`，不改写旧预测的误差货币。

## 反馈链路

全局 G 残差继续保留在诊断 `groups`，不再生成聚合或单例升降指令。请求边界同时过滤顶层 `prediction_reminders` 和 `strategy_outcome_calibration.reminders` 的旧 G 指令；C/R 提醒兼容性有回归覆盖。

`conditional_g` 只汇总相同指标、观测版本、执行范围及所列题族、结构格、求解条件和模型的事实。重复预测同一执行范围先去重，再分组；改变预测时点的预算快照不会增加执行数。报告通过、失败、未知、独立 episode 和带标签 episode 数，稀疏组只支持所列执行案例。Brier/分箱仅以同时有数值预测和可归因标签的单位为分母。

同题、同冻结原题、同验收标准和可比求解条件下，已执行的不同方法可以并列。显示行预算省略比较伙伴时，比较仍保留双方完整方法步骤和证据引用，并报告数量；未执行候选没有反事实标签。只有一种已执行方法时保留该事实。两个方法都通过同一二元 checker 时达标比较为平局；独立参考或目标重算核验过的业务目标可以作为案例事实保留。

## 实际 provider 请求片段

运行 `PYTHONPATH=src python references/examples/g_signal.py`：在临时库里走正常 predict → execute → record → check-task → close-episode，三次尝试分别产生 0、1、未知。以下从实际交给替身 provider 的下一次请求摘取字段，原值保留；长证据引用与完整反馈见 [g_signal_provider_request.json](g_signal_provider_request.json)。示例数值用于验证通路，不代表模型预测质量。

```json
{
  "benefit_convention": {
    "kind": "effective_completion",
    "metric": "task_check_passed_given_faithful_execution",
    "unit": "probability",
    "scope": "attempt",
    "observation_rule_version": "wm-g-faithful/1",
    "required": true
  },
  "conditional_g": [
    {
      "passed": 1,
      "failed": 1,
      "unknown": 1,
      "n_distinct_episodes": 3,
      "n_scored": 2,
      "brier": 0.29
    }
  ],
  "same_task_comparisons": [
    {
      "left": {
        "method_actual": {
          "n_steps": 1,
          "name": "separate selection",
          "steps": [
            "Select BE and CFC factories separately, then merge."
          ]
        },
        "observed": 0.0,
        "evaluation_id": "ev_2eb225d42e44"
      },
      "right": {
        "method_actual": {
          "n_steps": 1,
          "name": "joint selection",
          "steps": [
            "Choose a common factory for BE and CFC."
          ]
        },
        "observed": 1.0,
        "evaluation_id": "ev_6c4d0d37ca14"
      },
      "acceptance_comparison": "right_passed_left_failed"
    }
  ],
  "prediction_reminders": []
}
```

条件组在该请求中为通过 1、失败 1、未知 1、独立 episode 3；Brier 样本及分箱分母都是 2，Brier=0.29。显示的两个单行案例之外，同题比较仍携带通过方法和失败方法双方的来源引用。

## 回归验证

- 完整回归：`PYTHONPATH=src:tests/harness python -m unittest discover -s tests/harness`，**1730/1730 通过**（126.976 秒），日志 `/tmp/g-all-current.log`。
- 新 G 回归：`test_strategy_outcome.TestFaithfulGContract`、`test_wm_calibration_loop.TestFaithfulGObservations`、`test_calibration_v2.TestConditionalGFeedback`，**20/20 通过**（5.332 秒），日志 `/tmp/g-20-final.log`。
- G 及发布性能专项：上述相关案例和 `test_r22_publish_performance`，**29/29 通过**；完整回归也覆盖这些检查。
- 文档检查：`python references/examples/_check_docs.py` 通过；改动文档链接逐一检查通过。
- `python -m compileall -q src tests/harness references/examples` 通过；新隔离示例正常退出并生成实际 provider 请求片段。

HTTP 替身只监听本机；检查、求解和示例均使用临时工作区/隔离库。项目数据目录未参与验证。

新增 20 项测试（优先扩展指定三组测试），覆盖：新默认请求/响应/持久化/效用/结算贯通；solver-gap、1.2、非有限数、bool/字符串、错误单位/范围/观测版本不可用；缺值保留推演；忠实成功、方法遗漏、代码遗漏、真实超时、方法切换、代码修复、验收不足、证据不足、重述方法；旧无结果规则和旧误差口径；旧缓存双位置过滤、C/R 兼容；计数/发布幂等、未知分母、预算时点变化去重、同题不同方法和平局、稀疏支持及历史事实召回。旧 solver-gap 夹具改为显式声明旧口径，保留原有兼容性检查。

## 后续真实实验的验收指标

| 指标 | 记录口径 |
|---|---|
| 可归因标签覆盖率 | 有充分依据得到 0/1 的实际执行范围数 ÷ 实际执行范围总数；同一执行去重，并按未知原因、题族和范围分层 |
| 同题策略排序 | 同原题/验收/求解条件的真实不同方法配对；预测 G 排序与有效标签排序比较，二元同结果为平局，未知不强行排序 |
| 正确修订次数 | 主 agent 根据决定性条件修订方法，后续忠实执行并通过原题验收的次数；保留修订链接，代码修复与方法修订分开记录 |
| 无效求解次数 | 已运行但未得到可验收结果的真实求解尝试数，区分方法导致不达标、实现/环境无结果；验收不足单列未知，附成本 |

本轮证据证明机制接通。策略选择质量、正确修订增量和无效求解减少仍需后续真实对照实验；这里没有宣称提升。

## 仍缺观测依据的情况

旧记录未保存实际方法或代码/结果证据、只有步骤自述或单独 checker failed、验收覆盖不足、失败来源不清时，新 G 保持未知。代码偏离方案或换方法后的结果不归给原候选；后一次忠实执行可成为实际采用方法自己的案例。检查报告里的归因仍依赖有来源的审计，框架不自动证明数学模型完整。

## 改动文件

代码：

- [src/or_harness/api.py](../src/or_harness/api.py)
- [src/or_harness/strategy/verification.py](../src/or_harness/strategy/verification.py)
- [src/or_harness/world_model/strategy_prediction.py](../src/or_harness/world_model/strategy_prediction.py)
- [src/or_harness/world_model/contracts.py](../src/or_harness/world_model/contracts.py)
- [src/or_harness/world_model/episode_closeout.py](../src/or_harness/world_model/episode_closeout.py)
- [src/or_harness/world_model/g_signal.py](../src/or_harness/world_model/g_signal.py)
- [src/or_harness/world_model/context.py](../src/or_harness/world_model/context.py)
- [src/or_harness/world_model/conditions.py](../src/or_harness/world_model/conditions.py)
- [src/or_harness/world_model/planner.py](../src/or_harness/world_model/planner.py)

测试及隔离夹具：

- [tests/harness/test_strategy_outcome.py](../tests/harness/test_strategy_outcome.py)
- [tests/harness/test_r22_publish_performance.py](../tests/harness/test_r22_publish_performance.py)
- [tests/harness/test_task_check.py](../tests/harness/test_task_check.py)
- [tests/harness/g_fixtures.py](../tests/harness/g_fixtures.py)
- [tests/harness/test_wm_calibration_loop.py](../tests/harness/test_wm_calibration_loop.py)
- [tests/harness/test_linkage_attribution.py](../tests/harness/test_linkage_attribution.py)
- [tests/harness/test_capability_evolution.py](../tests/harness/test_capability_evolution.py)
- [tests/harness/test_r12_world_model.py](../tests/harness/test_r12_world_model.py)
- [tests/harness/test_command_surface.py](../tests/harness/test_command_surface.py)
- [tests/harness/test_world_model_m3.py](../tests/harness/test_world_model_m3.py)
- [tests/harness/test_hplus.py](../tests/harness/test_hplus.py)
- [tests/harness/test_calibration_v2.py](../tests/harness/test_calibration_v2.py)
- [tests/harness/test_episode_closeout.py](../tests/harness/test_episode_closeout.py)

运行时提示词文档、参考和请求产物：

- [references/prediction_context.md](../references/prediction_context.md)
- [references/world_model_contract.md](../references/world_model_contract.md)
- [references/episode_closeout.md](../references/episode_closeout.md)
- [references/strategy_outcome.md](../references/strategy_outcome.md)
- [references/commands.md](../references/commands.md)
- [references/examples/g_signal.py](../references/examples/g_signal.py)
- [SKILL.md](../SKILL.md)
- [docs/g_signal_provider_request.json](../docs/g_signal_provider_request.json)
- [docs/g_signal_validation.md](../docs/g_signal_validation.md)
