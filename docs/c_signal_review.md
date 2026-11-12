# C 信号两项 P1 修复

基线 `main@93b9cd22bf2c86bf1f8bcfccb4ef3a54f4028bd0`。已确认远端 HEAD 和本地源码一致，并保存修改前快照、保留其他本地文件。

## 1. 正常首提的 runner 信息

复现成立：先结算相同题族、宿主和初始阶段的历史，再对一个新任务直接预测，旧请求拿到 0 个范围、0 个案例，缺少宿主模型及阶段。

修复位置：`api.py` 构造/快照入口、`cli.py` 普通 runner 构造、`context.py` 快照投影和 `cost_signal.cost_conditions`。runner 可通过 `ORHarness(execution_context=...)`、`OR_EXECUTION_CONTEXT`（JSON/文件）或全局 `--execution-context` 传入实际 host/model、已知 phase 及来源。首提快照冻结这些值，无需先写 model/usage action。独立的世界模型配置不充当执行宿主；缺失信息继续未知。已有真实产物状态会推进阶段；缓存/历史上下文读取保存的值。

回归从头通过正常预测、执行、检查、结算构造历史。随后调用真实 CLI 的 `predict-strategy`：只替换网络 provider 和 embedding 实现，正常 `_harness`、参数解析、快照、历史发布读取均保留。新任务的动作数为 0，请求取得 1 个范围、1 个案例。另覆盖显式参数优先级、文件配置、缺失配置、阶段推进和冻结上下文不读取后续 runner 变化。

## 2. 用量冲突逐维保持未知

复现成立：旧版把“汇总 8 次工具调用、200 tokens + 1 条工具明细”结算为 1 次、0 tokens；把有执行器下界 1 的整窗 0 次也结算为有效实测。

修复位置：`cost_signal._call_values`、`scoped_host_observations`、新增执行器下界核对，以及 `usage.host_report_dimensions`。先按维度识别明细覆盖，再核对同窗口汇总/计数器/完整明细；部分明细只提供下界。执行器下界只取本窗口内的不同执行，重复引用不累加，重叠来源取最大下界。

| 输入 | 修复后的实测 |
| --- | --- |
| 汇总 tools=8、tokens=200；只有一条工具明细，未声明 tokens 覆盖 | tools 未知；保留完整 token 汇总 200，不再推断 0 |
| tokens 汇总 200、完整模型明细合计 100 | tokens 未知；合法 tools 汇总保留 |
| tools=0、执行器已证明至少 1 次 | tools 未知；其他合法维度保留 |
| 部分工具明细 1 条、汇总 tools=8 | 汇总可用；部分明细不冒充完整总量 |
| 部分工具明细已证明 2 次、汇总 tools=1 | tools 未知 |
| 汇总、去重完整明细及执行下界一致 | 合法实测继续统计 |
| 明确覆盖完整、tokens=0 且无模型调用 | 合法零值参与绝对/带符号误差，不进入 log-ratio |

冲突保留来源引用及原因，不能通过另一份报告补回该维度的标签。实测范围、案例和误差分母均排除冲突维度；G/R 及其他合法 C 维度仍可读取。原先用 tools=0 覆盖“合法零值”的回归会违反已知执行下界，现改用合法 tokens=0，并单独验证错误工具零值。

[请求与结算摘录](c_signal_review_examples.json) 来自替身 provider 和隔离库的真实调用链；只证明链路及校验规则，不是判别力实验。C 资源单位、`expected`/`expected_measured`、`wm-c-window/1`、效用、gate 和 G/R 契约沿用现有定义。

## 验证

新增 `tests/harness/test_c_signal_review.py` 的 18 个回归，扩展已有零值回归。最终结果：定向 **290/290** 通过（36.163 秒），`tests/harness` 全量 **1783/1783** 通过（140.749 秒）。


复验命令：

```bash
PYTHONPATH=src:tests/harness python -m unittest tests.harness.test_c_signal_review tests.harness.test_r13_remaining_cost tests.harness.test_cost_eligibility tests.harness.test_usage_accounting tests.harness.test_calibration_v2 tests.harness.test_wm_calibration_loop tests.harness.test_episode_closeout tests.harness.test_r19_solver_duplicate
PYTHONPATH=src:tests/harness python -m unittest discover -s tests/harness
```

改动共 13 个文件：`src/or_harness/{api,cli}.py`、`world_model/{context,cost_signal,usage}.py`，两份测试文件，`SKILL.md`、`references/{commands,prediction_context,episode_closeout}.md`，本文及请求/结算 JSON 摘录。其他本地修改及文件保留。
