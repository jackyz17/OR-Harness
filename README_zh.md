# OR-Harness（中文说明）

一个**小巧、可组合、对 harness 友好**的运筹优化（OR）策略学习能力层。

> 给定一个不断演进的大规模工业优化任务流，OR agent 能否从既往执行中学到"哪些求解策略适合哪些问题结构"，并以更低执行成本持续获得同等或更优的解质量？

OR-Harness 运行在外层 harness agent（Hermes 类）**内部**。它不是自治 agent：没有对话循环、没有隐藏全局状态、没有无人值守的后台工作。外层 agent 负责编排；本能力层提供建议、执行与记忆。

只有在你**显式配置**了 provider（`--world-model URL::MODEL`，或 `ORHarness(world_model=...)`）并**显式调用**预测命令时才会调用模型。未配置 provider 时，所有世界模型命令都返回明确的 `not_configured`，不发生任何网络活动。预测始终是影子假设：它不会变成事实，未执行候选的预测也不等于真实反馈。

**完整主流程需要 provider。** `execute` 可以在没有 provider 时求解，但那样就没有任何预测被生成、绑定或校准——不带 `--prediction` 的尝试没有可收尾的评价，所以缺 provider 是配置缺口，而不是降级模式。下面的快速开始因此走完整的预测路径；以 [SKILL.md](SKILL.md) 为权威走查。

## 核心能力

- **双层记忆**：有界的 **Execution Evidence Bank**（执行证据层：实际策略/实际质量/实际代价/失败与恢复/产物；只保留最近 N 个完整 episode 的滑动窗口，最旧的整集淘汰，绝不拆散修复链）+ 派生的 **Strategic Knowledge Bank**（战略知识层：期望质量/期望代价/期望风险,带预测区间与前瞻校验；来源引用过期后知识仍自足，凭自身内容与验证范围继续有效）。条件统计即时计算、永不落库。
- **确定性画像**：从任务 spec 或外层供给的 annotations 提取结构耦合特征——不建 NLP 子系统。
- **外层保有全部控制权的策略召回**：透明评分（`α·Q̂ − β·C_scalar − γ·R̂`）、双层证据回退、四消融模式。**框架内置策略目录已删除**：候选来自真实记忆（跑过或归纳过），由外层自行提出；冷启动时无先验分数——没有经验就如实返回空结果并说明原因，从零积累。
- **沙箱执行**：AST 策略 + POSIX rlimit + 墙钟超时；基本验证；五维代价计量（retries 计入代价）。
- **由你掌控的归纳**：`orx induction-material` 直接读取一批**已结束任务**的材料——成功、失败、跨格、跨方法名都可见，无需先命中候选、也不受样本数门槛限制——**新策略由你形成**：框架只报告证据**带有什么**（`method.basis`：是否有实际执行的方法？还是只有计划？还是一片空白），不替你打标签，也绝不从数字里推断方法。知识是**只增**的：每次 `induce --relation` 创建一条**新编号条目**（编号由框架分配），修订或相反结论都各自成为一条新条目，已有条目的核心内容**从不被改写或合并**。**发布由你决定**：框架只做管理性检查（格式合法、引用的执行存在、编号有效、写入成功），把你声明的 `check`/`--verify` 记为**你自己的审查轨迹**，不对内容设发布门槛、也不认证结论。空库会被如实报告为 `memory_state`，绝不是门槛。冷归档防复活。
- **结构化关系主张**（`induce --relation`）：跨任务知识——不是某个策略的统计量，而是"结构条件 → 建模/求解选择 → 后果"的关系，例如"时间耦合 ≥0.5 的调度任务上，时间分解必须保留跨期衔接状态"。每条主张显式引用**已记录的执行**并标注其在主张中的角色，可选地声明**可计算的部分**（`probe` / `status` / `comparison` 断言）——可写在关系载荷内嵌的 `check` 块，也可写在独立 `--verify` 载荷，**两者都会被读取**（同时提供时 `--verify` 优先并报告覆盖，绝不静默忽略）；证据身份（任务、家族、结构格、策略）由框架从事实派生，判定也被记录到条目上。**框架无法评估的谓词键绝不默认为满足**：召回会把它标为 `unknown` 并列出无法核验的键（与 `conflicts` 区分）。断言自带语义：`mode: paired` 只比较同任务配对，`mode: group` 比较双方在"每条记录都测得"的指标上的均值，`aggregation: all|mean` 决定不利样本如何处理。提交的主张即**被提供为知识**——只要编号有效、引用真实，它就进入召回；`verification` 块如实写明**检查了什么、没覆盖什么**（"在所声明范围内未发现违例"，绝不是"整段主张被证明为真"）。详见 [references/induction.md](references/induction.md)。
- **七个求解器适配器**（highs、pulp、ortools、scip、copt、pyomo、gurobi）——仅做可用性探测，具体求解器由你按情况选择。
- **统一世界模型契约**（`wm-contract/1`）：为两个预测模块提供有版本、可序列化、可校验的结构——**OR 策略后果预测**（收益携带指标/单位/基线，代价复用 `CostVector`，风险为具名事件，不确定性区分执行随机性与证据不足）与 **Harness 能力演化预测**（`H = F(M, W_OR, Pi, R, T)` 的能力证据、候选学习操作、基线与时间范围、学习代价、退化风险、验证条件）。构建契约本身不调用模型：它如实返回 `status="contract_only"`，并把 `provider_configured`（已挂载 provider）、`service_available`（本构建实现了该类服务）与 `prediction_made`（确实产生了预测）作为三个独立事实分别报告——仅配置了 provider 而模型调用次数为 0，永远不会是 `valid`。策略后果与能力演化两个服务都已实现（`wm-so/1` / `wm-ce/1`），agent 流程使用 `predict-strategy` / `plan-next` 与 `predict-capability`。旧无版本载荷通过显式 legacy 视图保持可读；未知契约版本明确失败，绝不猜测解析。详见 [references/world_model_contract.md](references/world_model_contract.md)。
- **冻结的预测输入上下文**（`wm-context/2`）：预测实际依据什么信息，以及这些信息如何一致、完整、可追溯地到达模型。一次 `orx context` 冻结一份上下文：**联合问题表征**（任务文本与载荷、CIR 关系——保留为关系而非压缩成三个数、带来源标注的数学属性、结构画像与推导报告）、**X/B**（来自同一快照）、**两条既有检索渠道的证据**（按证据身份去重，携带内容与适用性标注，跨格命中可见但不进入当前格统计）、**Harness 能力证据**（`H = F(M, W_OR, Pi, R, T)`，仅证据强度，无综合评分）、**外部执行约束**，以及自已结束 episode 回流的三个反馈块：聚合校准摘要、紧凑的逐条**预测—执行对照**（成功与失败都含、跨格案例可见、按预算限长、绝不静默裁剪成 top-k）与确定性**提醒**，外加 **H+ 跟进状态**。**建模前即可构建**：没有 `model`、没有 CIR、没有文本都是正常输入，缺失部分逐项如实报告，不从 `family` 名称臆断数学性质。构建上下文**不调用预测模型、不执行 solver、不做归纳**；同一次候选比较共享同一份上下文，复用需通过任务版本校验。详见 [references/prediction_context.md](references/prediction_context.md)。
- **策略后果预测服务**（`wm-so/1`）：基于冻结上下文的 training-free OR 策略后果预测服务。`orx predict-strategy` 预测单个候选的收益（带指标/单位/基线——`solution_quality` 必须是归一化值，原始目标值会被拒绝而非截断）、资源代价（`CostVector`，掩码标记的是预测维度）、风险（具名事件，与代价分开）与不确定性（模型自报，如实记录为未校准）；`orx plan-next` 在同一保守口径下比较候选（未知代价按峰值份额计、未知风险按满权重计、未知收益不计——这是决策规则而非实测概率）并给出建议；`orx bind-strategy` 将真实执行与所提配置的预测关联，带身份校验——未知身份字段（执行动作未记录 episode、动作日志不携带的配置键）单独记录为 unknown，绝不当作匹配。所有失败状态（未配置、provider 错误、空/非法载荷、越界数值）可区分并连同真实调用开销一并持久化。规划只使用该协议（不存在第二条预测路径）：旧 `predict_outcome` Python API 仅用于读取历史记录，CLI/agent 流程不再使用。详见 [references/strategy_outcome.md](references/strategy_outcome.md)。
- **Episode 收尾与经验校准**：`orx close-episode` 以诚实的终态（completed/failed/aborted/budget_exhausted）关闭一个 episode，对每个已绑定的策略后果预测做**逐字段**事后评价（收益误差用预测自己声明的指标/基线口径、代价逐维误差仅在双方都测量了同一范围时计算、有标签的风险事件记 Brier 分、区间覆盖——被排除的字段既不是命中也不是失误），并发布版本化的**经验校准摘要**供后续 episode 的新建预测上下文读取（仅统计已收尾 episode；低于样本下限如实报告 `insufficient_evidence`，绝不编造数值）。收尾幂等；不运行 solver、不调用模型、不触发归纳；冻结的预测永不改写，未执行的候选不会获得反事实标签。同一策略被再次选择是两个选择轮次（`round_index`），绝不聚合成一个评价样本。详见 [references/episode_closeout.md](references/episode_closeout.md)。
- **任务结果校验**：`orx check-task` 回答 `execute` 无法回答的问题——**这个答案是否满足原任务**。求解器的 `optimal + gap=0` 只说明"把自己的模型解好了"：一个用分数值回答整数问题的 LP 松弛同样会报 `optimal`，若不加校验就会作为成功样本进入召回、条件统计、世界模型反馈与离线归纳。校验基由你声明（`reference_objective`+`tolerance`、`reference_status`、`integer` 变量域、`recompute_objective` 目标重算、`semantic_probe` 逐值探针、`intent` 标记有意松弛/中间解），框架只评估声明过的部分，并把未检查项如实列入 `scope.unchecked`——目标匹配不证明模型正确，无 gold 不默认通过（`insufficient`，既不是 pass 也不是 fail）。确认不合格的答案**仍保留在证据集中**（成本真实计入、失败是反例材料），只是不再计入成功样本；它无法再因 `optimal + gap=0` 成为正样本。修复由外层 agent 完成：框架只提供原题、产物、检查报告与前次执行（`reflection_material`），不预设是建模错误、不要求重建、**严禁为匹配参考答案改变原任务**。`close-episode` 统一称"任务/episode 收尾"，收尾不等于答案正确；晚到的校验结果会改变后续使用（统计与校准即时采用），但不重写任何已存评价。

## 快速开始

```bash
pip install -e .                  # 零运行时依赖（纯 stdlib）
pip install -e ".[solvers-free]"   # 可选：highspy + pulp

orx doctor                        # 探测求解器、检查记忆目录
orx profile   --task t.json
orx recall    --task t.json --top 3 --candidate my-method --candidate alt-method
orx predict-strategy --task t.json --episode ep1 --candidate candidate.json
orx execute   --task t.json --episode ep1 --prediction sp_... --code solve.py --workspace ws
orx check-task exec_... --check '{"reference_objective": 10755, "integer": {"variables": ["x1", "x2"]}}'
orx record    --from-staged exec_... --usage-file usage.json
orx close-episode --task t.json --episode ep1 --terminal completed
orx calibration
```

每次尝试都需要 `--episode`，且预测必须在运行前绑定（`--prediction`）：`close-episode` 评价的正是这条关联。宿主的用量报告（`--usage-file`）才是真实 token 数；手填 `--override` 只是兜底，而"无来源的声明"必须加 `--override-source agent_estimate`，这样它永远不会被当作校准真值。

**归纳是独立的离线步骤，不属于第一轮。** `orx induction-material` 读取材料批；由你形成新策略后用 `orx induce --relation '{...}'` 提交——框架只写入你提交的内容，而可迁移策略需要**2 个不同任务**上观察到同一机制。等这些执行攒够之后、在 episode 之外再跑。

`my-method` 刻意是框架从未见过的名字：框架不内置任何策略目录，因此不存在"是否属于目录"这个问题。`recall` 只报告你点名的方案在真实记忆里有什么（初始状态下是"空 + 原因"），`predict-cost`/`execute` 接受你提出的任何方案。完整走查（空库 → 外层提出方案 → 代价未知 → 执行 → 答案校验 → 记录 → 召回）是一个可运行脚本：`PYTHONPATH=src python3 references/examples/no_catalog.py`。

记忆存放在显式目录（`--home` 或 `$OR_HARNESS_HOME`），底层为单个 SQLite 文件。

配置刻意保持显式——没有配置文件；全部配置面 = CLI 参数加环境变量：`$OR_HARNESS_HOME`（记忆位置）、`$OR_WM_API_KEY`（世界模型端点密钥；URL 与模型经 `--world-model URL::MODEL` 传入）、`$OR_WM_TIMEOUT` / `$OR_WM_MAX_TOKENS`（世界模型调用超时与输出预算，可被 `--wm-timeout` / `--wm-max-tokens` 覆盖）、`$OR_WM_NO_RESPONSE_FORMAT`（端点拒绝 JSON 模式时设为 `1`）、`$OR_WM_ENABLE_THINKING`（设为 `1` 时允许模型先思考再作答，默认关闭）、`$OR_EMBEDDING_BACKEND` / `$OR_EMBEDDING_BASE_URL` / `$OR_EMBEDDING_MODEL` / `$OR_EMBEDDING_API_KEY`（向量召回的嵌入后端）。其余都是单次调用参数——见 [references/commands.md](references/commands.md)（英文）。

## 文档

- **[SKILL.md](SKILL.md)** —— harness agent 的薄契约（从这里开始，英文）
- **[references/commands.md](references/commands.md)** —— 全部命令按工作阶段分组，并列出命令之间需要传递的 ID（英文）
- **[references/modeling.md](references/modeling.md)** —— 模型表征与 CIR：语法、约束标签、验证层次（英文）
- **[references/world_model_contract.md](references/world_model_contract.md)** —— 统一预测契约、`contract_only` 含义、attempt 与策略执行窗口之分、能力来源与迁移表（含可运行示例，英文）
- **[references/prediction_context.md](references/prediction_context.md)** —— 冻结的预测输入上下文：联合问题表征与数学属性来源、两条检索渠道与证据类别、能力证据强度、一次构建与一致复用规则（含可运行示例，英文）
- **[references/strategy_outcome.md](references/strategy_outcome.md)** —— 策略后果预测服务（wm-so/1）：协议、比较口径、预测—选择—执行关联（含可运行示例，英文）
- **[references/episode_closeout.md](references/episode_closeout.md)** —— Episode 收尾：真实结果摘要、逐字段事后评价、窗口选择轮次、经验校准通道（含可运行示例，英文）
- **[references/concepts.md](references/concepts.md)** —— 双层记忆、CostVector、派生层处置，以及三个必须分开的问题（求解器是否解好了模型 / 答案是否对任务有效 / 知识主张是否成立）（英文）
- **[references/induction.md](references/induction.md)** —— 四类归纳 pattern、适用范围=家族+结构格、入库门槛与入库验证、离线生命周期（英文）
- **[references/examples/task_check.py](references/examples/task_check.py)** —— 可运行：校验 → 诊断 → 修复 → 记录 → 收尾（松弛 LP 被整数域检查识别、修复后才被当作成功）（英文）
- **[references/examples/](references/examples/)** —— 其余可运行走查（`no_catalog.py`、`strategy_outcome.py`、`episode_closeout.py`、`capability_evolution.py`、`prediction_context.py`、`contract_roundtrip.py`，英文）

## 实验

```bash
PYTHONPATH=src python3 -m or_harness.experiments.runner
```

`experiments/runner.py` 在四消融模式（`none` / `cases` / `strategic` / `cost-aware`）下运行确定性合成任务流，输出逐任务 metrics CSV（质量、五维成本、累计成本、记忆规模）与 `summary.json`。C 与 D 的分离——质量持平、代价分化——是"代价感知是记忆必要成分"这一论点的实验支撑。

## 开发

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/harness -p "test_*.py"
```

纯 stdlib；求解器包为可选 extras，运行时探测。

## 许可

MIT
