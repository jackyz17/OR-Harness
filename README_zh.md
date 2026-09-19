# OR-Harness（中文说明）

一个**小巧、可组合、对 harness 友好**的运筹优化（OR）策略学习能力层。

> 给定一个不断演进的大规模工业优化任务流，OR agent 能否从既往执行中学到"哪些求解策略适合哪些问题结构"，并以更低执行成本持续获得同等或更优的解质量？

OR-Harness 运行在外层 harness agent（Hermes 类）**内部**。它不是自治 agent：没有对话循环、没有隐藏全局状态、没有无人值守的后台工作。外层 agent 负责编排；本能力层提供建议、执行与记忆。

只有在你**显式配置**了 provider（`--world-model URL::MODEL`，或 `ORHarness(world_model=...)`）并**显式调用**预测命令时才会调用模型。未配置 provider 时，所有世界模型命令都返回明确的 `not_configured`，不发生任何网络活动。预测始终是影子假设：它不会变成事实，未执行候选的预测也不等于真实反馈。

## 核心能力

- **双层记忆**：append-only 的 **Execution Evidence Bank**（执行证据层：实际策略/实际质量/实际代价/失败与恢复/产物；有损压缩暂缓）+ 派生的 **Strategic Knowledge Bank**（战略知识层：期望质量/期望代价/期望风险,带预测区间与前瞻校验；入库后有效性不依赖原始证据存活,可从保留证据重新归纳）。条件统计即时计算、永不落库。
- **确定性画像**：从任务 spec 或外层供给的 annotations 提取结构耦合特征——不建 NLP 子系统。
- **外层保有全部控制权的策略召回**：透明评分（`α·Q̂ − β·C_scalar − γ·R̂`）、双层证据回退、四消融模式。冷启动时无先验分数——没有经验就如实返回 `no_memory`，从零积累。
- **沙箱执行**：AST 策略 + POSIX rlimit + 墙钟超时；基本验证；五维代价计量（retries 计入代价）。
- **由你掌控的归纳**：每次 record 后 C1–C6 证据 hint；`induce` 永远是外层显式调用。经验适用范围 = 家族 + 支持证据所在的结构格（沿用旧版四区间，不用跨样本 min/max 跨度，避免把表现相反的区段合并）；建条目需要 ≥2 个不同任务的支持执行（重复同一任务不算复现）；**发布**需要入库验证通过（`induce --verify`：规则成立 / 修复有效 / 质量不降而代价下降），未验证候选只记录、不进推荐；冷归档防复活。触发准则不再引用先验分数，改为纯统计判据。
- **七个求解器适配器**（highs、pulp、ortools、scip、copt、pyomo、gurobi）——仅做可用性探测，具体求解器由你按情况选择。
- **统一世界模型契约**（`wm-contract/1`）：为两个预测模块提供有版本、可序列化、可校验的结构——**OR 策略后果预测**（收益携带指标/单位/基线，代价复用 `CostVector`，风险为具名事件，不确定性区分执行随机性与证据不足）与 **Harness 能力演化预测**（`H = F(M, W_OR, Pi, R, T)` 的能力证据、候选学习操作、基线与时间范围、学习代价、退化风险、验证条件）。**契约已实现，预测服务尚未接入**：构建出的契约会如实返回 `status="contract_only"`，而不是假装已经做过预测；并且把 `provider_configured`（已挂载 provider）、`service_available`（本构建实现了该类服务）与 `prediction_made`（确实产生了预测）作为三个独立事实分别报告——仅配置了 provider 而模型调用次数为 0，永远不会是 `valid`。旧无版本载荷通过显式 legacy 视图保持可读；未知契约版本明确失败，绝不猜测解析。详见 [references/world_model_contract.md](references/world_model_contract.md)。
- **冻结的预测输入上下文**（`wm-context/1`）：预测实际依据什么信息，以及这些信息如何一致、完整、可追溯地到达模型。一次 `orx context` 冻结一份上下文：**联合问题表征**（任务文本与载荷、CIR 关系——保留为关系而非压缩成三个数、带来源标注的数学属性、结构画像与推导报告）、**X/B**（来自同一快照）、**两条既有检索渠道的证据**（按证据身份去重，携带内容与适用性标注，跨格命中可见但不进入当前格统计）、**Harness 能力证据**（`H = F(M, W_OR, Pi, R, T)`，仅证据强度，无综合评分）与**外部执行约束**。**建模前即可构建**：没有 `model`、没有 CIR、没有文本都是正常输入，缺失部分逐项如实报告，不从 `family` 名称臆断数学性质。构建上下文**不调用预测模型、不执行 solver、不做归纳**；同一次候选比较共享同一份上下文，复用需通过任务版本校验。详见 [references/prediction_context.md](references/prediction_context.md)。
- **策略后果预测服务**（`wm-so/1`，世界模型 M3）：基于冻结上下文的 training-free OR 策略后果预测服务。`orx predict-strategy` 预测单个候选的收益（带指标/单位/基线——`solution_quality` 必须是归一化值，原始目标值会被拒绝而非截断）、资源代价（`CostVector`，掩码标记的是预测维度）、风险（具名事件，与代价分开）与不确定性（模型自报，如实记录为未校准）；`orx plan-next --protocol strategy-outcome` 在同一保守口径下比较候选（未知代价按峰值份额计、未知风险按满权重计、未知收益不计——这是决策规则而非实测概率）并给出建议；`orx bind-strategy` 将真实执行与所提配置的预测关联，带身份校验——未知身份字段（执行动作未记录 episode、动作日志不携带的配置键）单独记录为 unknown，绝不当作匹配。所有失败状态（未配置、provider 错误、空/非法载荷、越界数值）可区分并连同真实调用开销一并持久化。旧 `predict_outcome` 路径保持不变。详见 [references/strategy_outcome.md](references/strategy_outcome.md)。
- **Episode 收尾与经验校准**（世界模型 M4）：`orx close-episode` 以诚实的终态（completed/failed/aborted/budget_exhausted）关闭一个 episode，对每个已绑定的策略后果预测做**逐字段**事后评价（收益误差用预测自己声明的指标/基线口径、代价逐维误差仅在双方都测量了同一范围时计算、有标签的风险事件记 Brier 分、区间覆盖——被排除的字段既不是命中也不是失误），并发布版本化的**经验校准摘要**供后续 episode 的新建预测上下文读取（仅统计已收尾 episode；低于样本下限如实报告 `insufficient_evidence`，绝不编造数值）。收尾幂等；不运行 solver、不调用模型、不触发归纳；冻结的预测永不改写，未执行的候选不会获得反事实标签。同一策略被再次选择是两个选择轮次（`round_index`），绝不聚合成一个评价样本。详见 [references/episode_closeout.md](references/episode_closeout.md)。

## 快速开始

```bash
pip install -e .                  # 零运行时依赖（纯 stdlib）
pip install -e ".[solvers-free]"   # 可选：highspy + pulp

orx doctor                        # 探测求解器、检查记忆目录
orx profile   --task t.json
orx recall    --task t.json --top 3
orx execute   --task t.json --strategy S04 --code solve.py --workspace ws --solver highs
orx record    --execution exec.json --override llm_tokens=1840
orx induce    --strategy S04
orx inspect   --bank strategic
```

记忆存放在显式目录（`--home` 或 `$OR_HARNESS_HOME`），底层为单个 SQLite 文件。

## 文档

- **[SKILL.md](SKILL.md)** —— harness agent 的薄契约（从这里开始，英文）
- **[references/world_model_contract.md](references/world_model_contract.md)** —— 统一预测契约、`contract_only` 含义、attempt 与策略执行窗口之分、能力来源与迁移表（含可运行示例，英文）
- **[references/prediction_context.md](references/prediction_context.md)** —— 冻结的预测输入上下文：联合问题表征与数学属性来源、两条检索渠道与证据类别、能力证据强度、一次构建与一致复用规则（含可运行示例，英文）
- **[references/strategy_outcome.md](references/strategy_outcome.md)** —— 策略后果预测服务（wm-so/1）：协议、比较口径、预测—选择—执行关联（含可运行示例，英文）
- **[references/episode_closeout.md](references/episode_closeout.md)** —— Episode 收尾（世界模型 M4）：真实结果摘要、逐字段事后评价、窗口选择轮次、经验校准通道（含可运行示例，英文）
- **[references/concepts.md](references/concepts.md)** —— 双层记忆、CostVector、派生层处置（英文）
- **[references/induction.md](references/induction.md)** —— C1–C6、适用范围=家族+结构格、入库门槛与入库验证、离线生命周期（英文）
- **[references/examples.md](references/examples.md)** —— 四个完整走查实例（英文）

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
