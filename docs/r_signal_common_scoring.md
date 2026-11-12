# 单次技术 R 与 C/R 共同计分

审查基线：`main@1c74c3782dcceab35e5e59e5ed56407184f5d84f`。开始工作和提交前均核对远程 main；本轮在该基线上修改，保留其余文件。G 契约、C 的计量窗口、效用权重、归一化尺度和 gate 阈值沿用现有实现。

## R 预测与执行证据

R 固定为当前执行 agent、求解器接口和运行条件下，一次实现与执行尝试发生技术失败的概率。请求和预测 trace 保存 `implementation_failure / execution / attempt / wm-r-technical/1`。只有 `implementation_failure.probability` 可参与 R 计分，且必须是 [0,1] 的有限 JSON 数值；0 有效，缺失、布尔、字符串、非有限值、越界或重复事件保持未知。其他风险事件仍可报告和诊断。

| 标签 | 证据 |
|---|---|
| 1 | 明确的代码构造/API 异常、异常退出、结果交付协议错误。例如声明 optimal 但未给目标值 |
| 0 | 单次流程正常完成并按协议交付可读结果，包括合法 infeasible；捕获异常后成功交付也可为 0 |
| unknown | 无可归因执行、环境阻断、普通限时或失败原因不能确认 |

回执保存 executed、退出码、异常类型、结果是否收到、协议问题、来源和原因。`ValueError: invalid network flow indices` 按捕获的异常类型处理，不因错误文字包含 network 被误归环境。代码导致的超时只有在已有记录附带有来源的调查、且引用该次代码及捕获轨迹/结果产物时才记 1。业务语义偏离继续留在 G 证据中。失败尝试与修复尝试各自绑定；G 的策略窗口不会把多个尝试压成一个 R 标签。

新版概率只与新版标签评分；旧预测保留旧观测适配器。新版 Brier 审计键为 `implementation_failure@wm-r-technical/1`。真实标签按执行去重，原始概率不先舍入，未知标签不进入 Brier。统计分别报告实际尝试数、已知标签数、已报概率数、可评分数和独立 episode 支持数。

## 已有发布窗口内的轻量参照

`close-episode` 从现有执行记录发布 `technical_r`，包括未做预测、未报 R 的可信执行。按实际执行 agent（host/model/可选 agent_id）、实际记录的 solver interface 和粗结构分组。正常 runner 在首次预测前通过既有 execution_context 或 OR_EXECUTION_CONTEXT 传入已知身份和接口，执行回执保存实际身份；身份/接口缺失不会借用其他 agent 的基率。

下一次请求的 `implementation_r_evidence` 在体积裁剪前选取最多三个适用分组，只展示失败/成功/未知、支持范围、执行引用及少量已跑通脚本产物事实。完整统计和 Brier 留在审计汇总；旧分类器的实现失败率不会成为新 R 的先验。开放 episode、结算后的离线执行和冻结决策之后才发布的结果不会进入该决策历史。

实际通过普通 API 发给替身 provider 的片段、对应结算和计划评分见 [r_signal_common_examples.json](r_signal_common_examples.json)。其中 5 次隔离执行得到 4 个已知标签（2 次失败、2 次成功）和 1 个未知；4 个已报概率中只有 3 个可评分。未预测的正常执行计入成功事实。下一条新任务首提请求收到一个匹配 agent/interface 的简短分组。

## C/R 共同计分

先确定真正进入同一排名的可比 G 候选，再确定共同项。不可排名候选不改变他人的成本尺度、维度或 R 是否计入。

| 输入 | 实际计分 |
|---|---|
| A、B 扣风险前均 .50，R 分别 .20、未知 | 两者均跳过 R，仍为 .50；A 原始 .20 保留 |
| R=.20，同时诊断超时概率=.80 | R 只取 .20 |
| A=500k tokens/120 秒，B=700k tokens/未知时间 | 两者只用 tokens；既有峰值尺度给出 C=5/7、1，未知时延保留 |
| 每个候选都有有效 R 和 C，含显式 0 | 使用共同 G/C/R，0 保持有效 |
| 无共同成本维度，R 也缺失 | C 不可比，全组按 G 计分；记录为部分效用 |
| A 有端到端时间及 solver time，B 只有 solver time | 先求交集，再去重，因此共同计 solver time |

C 的共同集合来自所有候选有效且同口径的 `expected_measured` 维度交集，再统一应用成本组件去重。attempt sandbox latency 与 remaining latency 保留各自范围。所有候选共用同一套现有权重和尺度。`scoring_basis` 保存参与候选、实际项、维度、范围、尺度、权重和缺失原因；`risk_probability`、`risk_reported`、`cost_reported` 保留原始可读输入。

## 修改位置

- `world_model/strategy_prediction.py`、`contracts.py`：运行时 R 提示词、固定约定、严格解析及缺失处理。
- `execution/executor.py`、`strategy/triggers.py`：技术结果回执、协议校验、异常类型优先分类。
- 新增 `world_model/r_signal.py`，并接入 `episode_closeout.py`：单次归因、独立观测版本、去重统计和历史选取。
- `api.py`、`world_model/context.py`、`cost_signal.py`、`conditions.py`：runner 身份转交、冻结请求、适用参照和旧口径隔离；C 用量计算未改。
- `world_model/planner.py`：C/R 共同依据及部分效用审计。
- 新增 `tests/harness/test_r_technical.py`；同步现有 strategy outcome、episode/calibration、r12、M3 回归断言。两项原有毫秒级预算测试改用受控时钟，生产预算逻辑未改。
- `SKILL.md` 及 `references/strategy_outcome.md`、`world_model_contract.md`、`episode_closeout.md`、`prediction_context.md`、`commands.md` 同步运行定义。

## 验证与边界

验证结果：定向回归 **231 项通过**；`tests/harness` 全量 **1810 项通过**，本轮新增 27 项测试方法。定向覆盖 strategy outcome、episode closeout、calibration v2、r12、M3 及新技术 R 测试。命令如下（HTTP 替身使用本机回环端口）：

```bash
PYTHONPATH=src:tests/harness python -m unittest \
  tests.harness.test_r_technical tests.harness.test_strategy_outcome \
  tests.harness.test_episode_closeout tests.harness.test_calibration_v2 \
  tests.harness.test_r12_world_model tests.harness.test_world_model_m3
PYTHONPATH=src:tests/harness python -m unittest discover -s tests/harness
```

回归经过真实的 predict、execute、record、close-episode、下一次 provider 请求及 plan-next 入口，provider 和宿主身份是明示的替身，库均隔离。既有断言已同步新的共同缺项规则；两项受机器调度影响的毫秒级时限测试使用受控时钟，继续断言调用次数、超时状态及已花费成本。

当前工作区运行库没有已记录执行或已关闭 episode；未提供可评估的真实配对样本。这里验证了输入、观测、反馈和计分链路，没有据此宣称 R 的真实判别力或策略选择收益。后续结论需要同口径、有来源、可归因的实际执行样本；正常限时、外部阻断、缺少 agent/interface 和仅有旧错误类的记录仍可能未知。
