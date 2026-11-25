# build6.3：文件驱动执行与失败观测验收

日期：2026-11-25。基线：`f22077905e65d03cc76631464e76227514e7c9bd`，已通过 GitHub 连接器确认是当时的远端 main。实现沿用现有 Executor、plan-next、execute、Execution Evidence 和 closeout，没有新增并行执行架构。

文件驱动策略已能直接读取原始文件，在同一个已绑定预测的正式 attempt 中完成复合求解并留下可核验的证据。全量测试仍有基线既有问题，不能称为全量全绿；具体失败集合和执行命令保存在 [机器可读验收记录](build63_regression.json)。

## 原因与输入接口

原来的 `SafePythonExecutor.validate_source` 只接受字符串字面量形式的 `open`，且只放行相对的 `result.json`；`from pathlib import Path` 也被直接拒绝。任务没有对应的输入授权清单。AST 既误拒合法数据读取，也不能对第三方库和原生读取接口统一执行文件边界；原先的 cwd、rlimit 和静态检查没有提供这样的进程级文件隔离。

在原 task JSON 中增加可选的 `input_resources`，不增加执行命令：

```json
{
  "task_id":"network-example", "family":"routing",
  "text":"按原题要求求解声明的网络实例。",
  "input_resources":[
    {"source":"data/nodes.csv","path":"network/nodes.csv"},
    {"source":"data/edges.csv","path":"network/edges.csv"},
    {"source":"data/config.json","path":"config.json"}
  ]
}
```

CLI 的 `source` 相对 task JSON 文件解析，Python API 相对调用者 cwd；允许明确的绝对来源。`path` 是 `inputs/` 下唯一的 POSIX 相对文件名，可以包含子目录。只接受显式声明的普通文件，不展开目录，不自动扫描附件、工作目录或 benchmark 目录。可信 Runner 必须保证清单只含本题公开输入，不含答案、隐藏 checker、其他任务或记忆库。纯文本任务不需要空清单。

准备阶段流式计算完整 SHA-256，将内容原子发布到现有的内容寻址 artifact 存储，按内容去重。预测、任务身份、执行 action、execution 和自检材料保留来源、相对路径、大小、哈希及 artifact 引用，不把大文件正文放进模型上下文，也不为每条 Evidence 再复制原始文件。提交时复核原文件及缓存内容；声明变化要求重新预测，提交时缺失、变化或损坏会留下已绑定的失败记录。

正常脚本可使用 `open("inputs/network/nodes.csv")`、`Path("inputs") / filename`、`Path.open/read_text`、`json.load`、`csv`，以及已安装的 Pandas、NumPy 和原生求解器文件接口。输入格式不限于 CSV/JSON/TXT；普通实例文件使用对应的已安装解析器。动态路径不再仅因为不是字面量而被拒绝。

正式命令仍为：

```bash
orx execute --task task.json --prediction P --code ws/solve.py --workspace ws
```

脚本在新的隔离 `/work` 中运行，读取 `inputs/<path>`，写入 `result.json`，临时文件放在 `scratch/`。只有本次 `result.json` 导出到宿主 `ws/result.json`。宿主 workspace 的其他既有文件不进入执行环境。配置、方法回执继续使用 `OR_ACTION_ID`。

## 安全边界与限制

采用 Linux 用户、挂载、网络命名空间、chroot、只读绑定挂载和 seccomp。保留危险模块和调用的 AST 检查；Python 审计提供路径错误诊断，真正的只读和访问边界也覆盖绕过 Python `open` 的原生接口。输入、脚本、运行时目录及其余 jail 目录只读；仅结果文件、日志描述符和 64 MiB 总量的 scratch 可写。来源各级目录禁止符号链接；路径穿越、未声明文件、越界绝对路径、其他任务和记忆文件均未获授权。

网络/socket、独立子进程/exec、命名空间和挂载修改、跨进程访问被禁止；数值计算线程仍可使用。保留 CPU、2 GiB 地址空间、64 MiB 单文件、正式 wall-clock 和进程组清理限制；日志改为受文件限制的有界采集，避免无界 PIPE 耗尽父进程内存。BLAS/OMP 默认单线程，求解器自己的线程和时间限制仍应显式配置并回读。

需要 Linux 的用户/挂载/网络命名空间、`unshare`、`libseccomp.so.2`；不具备隔离能力时拒绝执行，没有退回 AST-only 的路径。允许的只读运行时包括 Python/系统库、当前进程的 maps/可执行文件链接、CPU 信息，以及必要的 null/urandom；没有挂载整个宿主 `/proc`。已知任务、workspace 或 memory 与运行时目录重叠会被拒绝。

边界信任宿主内核、已安装运行时及其目录内容，不是 VM，也不提供文件元数据保密或抵抗并发宿主文件篡改的保证。Runner 必须把其他任务和敏感材料放在可信运行时目录之外。需要二进制子进程或远端许可证服务的后端仍受原有子进程/网络限制；不能把模块已安装解释为全部求解能力已验证。

## 执行流程、诊断与信号

准备阶段允许查看数据、表头/规模/类型/索引、数学结构、阅读证据、写代码及极小人工实例的 API 确认。原题 LP 松弛、子问题、搜索、列筛选、优化驱动的约简、多求解器比较和真实实例代码试错进入已选策略的正式 execute。一个候选可包含完整的 LP → 界 → 约简 → MILP，不要求每个子步骤分别预测。禁止工作流把宿主预求解后的答案交给 execute 单纯重放；框架没有新增宿主终端监控，行为审计仍由 Runner 承担。

保留执行后的可行性检查、目标重算、证书及针对具体疑问的独立验证。重要脚本/结果走已有 verification/artifact 通道，成本通过 verify action 或 host usage 记录。产生新方法或实质修订答案时重新 plan-next；没有增加未绑定的隐式求解重试。最大重试指导、单候选预测、候选绑定、H+ 硬门控及离线发布流程均沿用原机制。

状态仍使用原来的结果状态、`error`、`timeout`。`execution_features.technical_outcome` 增加具体诊断：`failure_stage`、`failure_cause`、`process_started`、`submission_rejected`、`runtime_eligible`，同时保留 `executed`、`result_received`、退出码及协议问题。

| 情形 | 阶段及观测 |
|---|---|
| 语法/明确脚本契约违规 | submission；未启动脚本；implementation_failure |
| 输入缺失、内容变化、缓存损坏 | submission；保留具体原因；责任不明确时为 Unknown |
| 无法启动进程 | launch；environment_failure；没有求解耗时 |
| 隔离建立失败 | sandbox_setup；进程已启动但脚本未启动；environment_failure |
| Python/API 异常 | runtime；实现错误与依赖/许可证不可用分别归因 |
| 非法结果/协议 | result；implementation_failure；保留可用原始产物 |
| Harness wall-clock 中断 | timeout；若尚在隔离建立阶段，runtime_eligible=false；若脚本已启动，则 true |
| 求解器内部限时并正常返回结果 | 不自动等于 Harness timeout |

提交拒绝一般保留 formal timeout=0，但不能作为算法实际避免超时的证据。隔离建立阶段确实被正式 deadline 中断则 formal timeout=1，同样不冒充算法运行证据。`technical_r.runtime_coverage` 单独报告启动资格，不改变原 formal timeout/Brier 的定义。未启动脚本时只记录真实提交/启动开销，不伪造 solver runtime。失败仍保留预测、候选、action/execution、提交脚本、输入引用、失败原因、成本及 H+ 审查材料；已消耗的预测不被释放为隐式重试。

没有改变 G/C/R/H+ 信号架构、R 事件集合及最大项聚合、C 维度/权重/归一化、U 权重、检索阈值或 H+ 核心提示词/门控。仅澄清 G 与纯技术 R 的边界，并依据新结构化证据修正当前失败归因。旧记录不迁移、不改写历史标签；旧预测的评分口径有专门回归检查。数学模型正确不自动意味着 G=1，限时近似正常完成后的解质量仍可影响 G。

`doctor` 和预测的 `execution_constraints` 复用求解器适配器信息，提供入口 API、变量域、约束族、内部限时用法及执行限制。仅向预测发送已发现模块的提示。模块发现、后端/许可证、沙箱实用性分开表达，后两者未确认时为 unknown；不再在普通能力查询中重复启动 Gurobi 许可证探测。

## 验证结果

| 检查 | 实测结果 |
|---|---|
| 最终定向验收：文件/安全/失败、R/G/C/H+、历史口径、绑定、方法证据、提示词传递 | 394 项，成功；2 项可选依赖测试在默认解释器跳过，另行补跑 |
| NumPy/Pandas/SciPy 环境的最终文件与失败验收 | 29/29 成功，无跳过 |
| CLI 跨进程及端到端 | 13/13 成功，允许临时 localhost fixture |
| 最终显式原题的 LP → reduced-cost fixing → MILP 闭环 | 成功：LP 下界 1.5，安全固定 1 列，原问题最优目标 2；G 归因及 H+ 审查派发成功 |
| 全量对照 | 基线 2102 项，本轮 2130 项；两者均 22 failure、6 error，具体失败集合完全一致；本轮 2 项可选依赖测试另行补跑 |
| 文档命令/符号/链接、compileall、diff whitespace | 成功 |

全量对照先于最后补充的隔离建立阶段 deadline fixture；该追加分支由最终 394 项定向验收及 29 项带依赖验收覆盖。没有通过提高预算、删失败测试或改 H+/G/C/U 架构来掩盖基线问题。

原始 main 的归档测试辅助器还错误恢复了 `ActionLog._decode` 的方法绑定，导致后续数百项连锁错误；本轮只修正该测试辅助器。为公平对照，在基线副本也应用相同的测试辅助器修正，生产实现保持 f220779。22/6 的既有问题包括旧 CLI capability 检查、快照旧 golden、部分旧评分/规则生命周期断言等；完整测试 ID 见 JSON。

文件验收覆盖单 CSV、关联 CSV/JSON、TXT/实例文件、子目录、动态路径、Path/Pandas/NumPy、纯文本和超过 1 MiB 的未内联输入；安全验收覆盖未声明/其他任务/记忆文件、输入写入及原生 truncate、穿越/绝对路径/符号链接、第三方及原生读取绕过、网络/exec/fork、输出范围、scratch 总量、线程与超时。受控故障 fixture 只证明观测和记录机制，不是世界模型真实风险预测能力的证据。安全测试也在不依赖外层网络沙箱的环境下运行。

在可访问的本地材料中未找到 `p200x1188c`、`nw04` 原始文件，因此未执行真实工业实例回归。使用独立构造的文件驱动集合覆盖实例，原题明确二元域、覆盖需求、费用目标及最优性要求；求解脚本不含任何历史 benchmark 最优值。

## Runner 配合及未完成事项

Runner 继续负责任务级硬停止、session write lock、会话生命周期/调度/主动中断、公开输入授权及其来源、宿主终端原题预求解审计、跨题额外笔记隔离，以及分析/探查/编码/验证的完整使用成本上报。框架无法从代码层完全禁止宿主预求解。

尚未完成：真实 p200x1188c/nw04 实例回归；七类适配器在实际部署许可证/后端下的逐一运行确认；既有 22/6 全量基线问题。非 Linux 或禁用命名空间的部署需先配置可靠隔离能力，否则正式执行会明确拒绝。复杂工业实例的求解性能与解质量没有由小型集成 fixture 推广得到保证。

## 全部修改文件

文件清单（含本报告和 JSON）由 Git 相对基线差异生成，保存在 [build63_regression.json](build63_regression.json) 的 `changed_files` 中；以下按交付范围列全：

- `src/or_harness/execution/inputs.py`
- `src/or_harness/execution/sandbox.py`
- `src/or_harness/execution/executor.py`
- `src/or_harness/api.py`
- `src/or_harness/cli.py`
- `src/or_harness/adapters/solver.py`
- `src/or_harness/strategy/triggers.py`
- `src/or_harness/world_model/context.py`
- `src/or_harness/world_model/state.py`
- `src/or_harness/world_model/r_signal.py`
- `src/or_harness/world_model/episode_closeout.py`
- `src/or_harness/world_model/contracts.py`
- `src/or_harness/world_model/strategy_prediction.py`
- `tests/harness/test_file_execution.py`
- `tests/harness/test_execution_observations.py`
- `tests/harness/test_executor.py`
- `tests/harness/test_r_technical.py`
- `tests/harness/test_calibration_v2.py`
- `tests/harness/test_r24_archive_scan.py`
- `tests/harness/test_r_prediction_reasoning.py`
- `SKILL.md`
- `README.md`
- `README_zh.md`
- `references/commands.md`
- `references/self_check.md`
- `references/world_model_contract.md`
- `references/episode_closeout.md`
- `docs/validation/build63_executor.md`
- `docs/validation/build63_regression.json`
