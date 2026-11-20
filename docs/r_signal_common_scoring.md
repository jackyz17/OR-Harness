# 单次技术 R 与 C/R 共同计分

审查基线：`main@814deda`（codex 已合入“移除执行身份”改动）。本轮在其上把 R 从单一 `implementation_failure` 扩展为三项事件的最大单项风险分数，保留其余文件。G 契约、C 的计量窗口、效用权重 U = alpha*G - beta*C - gamma*R、归一化尺度和 gate 阈值沿用现有实现。

## R 预测与执行证据

R 是“最大单项风险分数”：在**所有参与排名的候选**都给出有效概率的事件集合 E 上取最大值，`R = max(p(event))`。可参与 U 的 R 事件固定为三项：`implementation_failure`（harness 自身代码/API 或结果交付技术失败）、`timeout`（执行被 harness 墙钟上限**中断**且未交付可用结果；求解器在自己的限时内正常返回可用结果不算该事件）、`environment_failure`（依赖、许可证、权限、沙箱策略或外部服务/后端阻断；普通代码错误不算该事件）。R 是最大单项风险分数，**不是联合失败概率**，也不对合成最大值计算 Brier。每个概率必须是 [0,1] 的有限 JSON 数值；0 是有效预测。未知不补 0：任一候选缺失/非法的事件会从**全组**的 E 中移除；E 为空、或候选声明了不同/未知的 R 计分口径时，R 对全组省略并写明原因。重复事件与非法概率把该事件标为不可用（unknown），不静默转换。请求和预测 trace 保存 `events / execution / attempt / wm-r-max-event/1`（实现标签仍用 `wm-r-technical/1`），旧预测保留原单事件 `wm-r-technical/1` 口径，不回填新概率。其余风险事件继续诊断，不参与本轮 U。

| 标签 | 证据 |
|---|---|
| `implementation_failure` 1 | 明确的代码构造/API 异常、异常退出、结果交付协议错误。例如声明 optimal 但未给目标值 |
| `implementation_failure` 0 | 单次流程正常完成并按协议交付可读结果，包括合法 infeasible；捕获异常后成功交付也可为 0 |
| `implementation_failure` unknown | 无可归因执行、环境阻断、普通限时或失败原因不能确认 |
| `timeout` 1 | harness 墙钟上限中断执行，未交付可用结果 |
| `timeout` 0 | 未触发墙钟中断；求解器在自己的限时内返回了可读结果（即使其 status 写 timeout） |
| `timeout` unknown | 旧记录无法区分“被中断”与“正常限时返回”时不猜 |
| `environment_failure` 1 | executor 记录的 `error_class == 'environment'`（依赖、许可证、权限、外部服务/沙箱策略） |
| `environment_failure` 0 | 执行未记录环境阻断类失败 |
| `environment_failure` unknown | 失败没有 error_class，或仅记录 unknown/label 不可确认时不推断 |

回执保存 executed、退出码、异常类型、结果是否收到、协议问题、来源和原因；同一份绑定的 attempt 事实同时给出三项事件的 label/basis。`ValueError: invalid network flow indices` 按捕获的异常类型处理，不因错误文字包含 network 被误归环境。代码导致的 `implementation_failure` 超时判定沿用原有有来源调查规则，且与新的 `timeout` 中断事件分开：前者是“代码原因导致的失败”，后者是“执行被墙钟中断”。业务语义偏离继续留在 G 证据中。失败尝试与修复尝试各自绑定；G 的策略窗口不会把多个尝试压成一个 R 标签，后续重试不进入本轮校准。

新版概率只与新版标签评分；旧预测保留旧观测适配器。`implementation_failure` 的 Brier 审计键为 `implementation_failure@wm-r-technical/1`；`timeout` 的新规则样本带 `timeout@wm-events/4`，旧 `wm-events/3` 样本不被合并。真实标签按执行去重，原始概率不先舍入，未知标签不进入 Brier。`technical_r` 分别报告三项事件的实际尝试数、已知标签数、已报概率数、可评分数和独立 episode 支持数（`coverage_by_event` / `events`），并明确 `aggregation`：R 取共同事件最大值，**不对 max/联合事件计算 Brier**。

## 已有发布窗口内的轻量参照

`close-episode` 从现有执行记录发布 `technical_r`，包括未做预测、未报 R 的可信执行。按实际记录的 solver interface 和粗结构分组。执行 agent 的一致性由调用方管理，框架不采集执行身份用于分组或筛选。`execution_context` 或 `OR_EXECUTION_CONTEXT` 仅声明已知求解器接口与建模阶段。

下一次请求的 `implementation_r_evidence` 在体积裁剪前选取最多三个适用分组，展示三项事件各自的 occurred/not_occurred/unknown 事实、支持范围、执行引用及少量已跑通脚本产物事实（超时与环境故障事实随历史一起可见）。完整统计和 Brier 留在审计汇总；旧分类器的实现失败率不会成为新 R 的先验。开放 episode、结算后的离线执行和冻结决策之后才发布的结果不会进入该决策历史。

实际通过普通 API 发给替身 provider 的片段、对应结算和计划评分见 [r_signal_common_examples.json](r_signal_common_examples.json)。其中 5 次隔离执行得到 4 个已知标签（2 次失败、2 次成功）和 1 个未知；4 个已报概率中只有 3 个可评分。未预测的正常执行计入成功事实。下一条新任务首提请求收到一个匹配 solver interface 的简短分组。

## C/R 共同计分

先确定真正进入同一排名的可比 G 候选，再确定共同项。不可排名候选不改变他人的成本尺度、维度或 R 是否计入。

| 输入 | 实际计分 |
|---|---|
| A、B 扣风险前均 .50，R 分别 .20、未知 | 两者均跳过 R，仍为 .50；A 原始 .20 保留 |
| R=.20，同时 timeout=.80，两者在全部候选有效 | R = max(.20,.80) = .80；若另一候选未给 timeout，则 timeout 进入“共同缺失”，R 仍为 .20 |
| A=500k tokens/120 秒，B=700k tokens/未知时间 | 两者只用 tokens；既有峰值尺度给出 C=5/7、1，未知时延保留 |
| 每个候选都有有效 R 和 C，含显式 0 | 使用共同 G/C/R，0 保持有效 |
| 无共同成本维度，R 也缺失（E 为空） | C 不可比，全组按 G 计分；记录为部分效用与遗漏原因 |
| 旧预测（`wm-r-technical/1`）与新预测混排 | 口径不同，R 对全组省略并注明原因，不混算新旧含义 |
| A 有端到端时间及 solver time，B 只有 solver time | 先求交集，再去重，因此共同计 solver time |

R 的共同集合来自参与排名候选的有效事件交集；未知/非法/重复事件对全组移除，E 为空或口径不一致时 R 全组省略。C 的共同集合来自所有候选有效且同口径的 `expected_measured` 维度交集，再统一应用成本组件去重。attempt sandbox latency 与 remaining latency 保留各自范围。所有候选共用同一套现有权重和尺度。`scoring_basis` 保存参与候选、实际项、维度、范围、尺度、权重、缺失原因，以及 R 的 `risk_events`、`risk_event_probabilities`、`risk_max_source`、`omitted_risk_events`（含每个事件的省略原因）和 `risk_interpretation`；`risk_probability`（旧单事件诊断值）、`risk_reported`、`cost_reported` 保留原始可读输入。

## 修改位置

- `world_model/contracts.py`：`R_EVENTS` / 三事件定义 / `wm-r-max-event/1` 计分口径 / 旧单事件口径。
- `world_model/strategy_prediction.py`：运行时 R 提示词（三项显式考虑、max 语义、未知/0/重复/非法处置）、请求 `risk_convention` 与严格解析。
- `execution/executor.py`、`strategy/triggers.py`：技术结果回执、协议校验、异常类型优先分类。
- 新增 `world_model/r_signal.py`，并接入 `episode_closeout.py`：单次归因、独立观测版本、去重统计和历史选取。
- `api.py`、`world_model/context.py`、`cost_signal.py`、`conditions.py`：codex 已移除执行身份；本轮保留其现状，不恢复求解 agent 身份机制。
- `world_model/planner.py`：三项事件共同集合 E、`R = max` 及最大风险来源/遗漏审计；U 权重与 G/C/H+ 未改。
- `world_model/episode_closeout.py`：`wm-events/4` 的 timeout 中断判定、三项事件绑定的 attempt 观测（`r_events_observation`）、逐事件覆盖率与 Brier。
- 扩展现有 `tests/harness/test_r_technical.py`（提示词/解析/共同 E/max/未知与 0/重复与非法/旧口径/逐事件校准/证据块），并同步 `test_calibration_v2.py`、`test_episode_closeout.py`、`test_r23_budget_labels.py` 的版本与逐事件键断言。
- `references/concepts.md`、`world_model_contract.md`、`strategy_outcome.md`、`episode_closeout.md` 与本文件同步运行定义。

## 验证与边界

本轮按要求只跑与本轮 R 扩展直接相关的定向回归：不跑全量、不建库、不调用真实模型。命令如下（HTTP 替身使用本机回环端口，`TMPDIR` 指向工作区内临时目录）：

```bash
TMPDIR=$PWD/.tmp_tests PYTHONPATH=src:tests/harness python -m unittest \
  tests.harness.test_r_technical tests.harness.test_calibration_v2 \
  tests.harness.test_episode_closeout tests.harness.test_r23_budget_labels \
  tests.harness.test_world_model_m3 tests.harness.test_wm_calibration_loop \
  tests.harness.test_r12_world_model
```

- `test_r_technical` 55 项通过：提示词/解析、共同事件集合 E、`R = max`、未知与显式 0、重复与非法概率、旧口径保留、逐事件校准与覆盖率、证据块。
- `test_calibration_v2` 63 项通过：occurrence 与 prediction-pair 两种计数单位分离、逐事件 Brier 键、覆盖率与版本隔离。
- `test_episode_closeout` 49 项通过；`test_r23_budget_labels` 通过（口径版本断言更新为 `wm-events/4`）；`test_world_model_m3` 与 `test_wm_calibration_loop` 通过。
- `test_r12_world_model` 仅剩 1 个与本轮无关的既有失败：`_compact_method_steps` 在拉取的基线 `814deda` 上已是“保留全部步骤”的桩实现，测试仍断言旧的中段省略标记；在未改动的基线上可复现，非本轮引入。
- `test_documented_result_keys` 的 CLI 参数错误与 `test_r24_archive_scan` 的 monkeypatch 错误同样在基线上复现，未纳入本轮定向集。

回归经过真实的 predict、execute、record、close-episode、下一次 provider 请求及 plan-next 入口，provider 是明示替身，库均隔离。timeout 用例用真实沙箱墙钟中断（1 秒）与“求解器正常限时返回结果”两种记录分别验证标签；旧预测保留 `wm-r-technical/1` 单事件口径，不回填新概率。

当前工作区运行库没有已记录执行或已关闭 episode；未提供可评估的真实配对样本。这里验证了输入、观测、反馈和计分链路，没有据此宣称 R 的真实判别力或策略选择收益。后续结论需要同口径、有来源、可归因的实际执行样本；旧记录无法区分“中断/正常限时返回”、缺少接口和仅有旧错误类的记录仍可能未知。
