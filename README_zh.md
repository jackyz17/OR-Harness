# OR Harness（中文版）

一个面向 LLM Harness Agent 的 Python 框架，实现**多求解器并行探索求解 OR 问题**、**积累经验证的求解经验**、以及离线地**从历史经验中归纳出新的优化原则**（"举一反三"）。

 [English](README.md)

---

## 快速上手

```bash
# 求解 OR 问题（mock 演示，无需装求解器/配置 LLM）
python3 scripts/or_experience_cli.py solve --mock-demo --problem "Assign tasks to machines to minimize cost" --json

# 完整求解演示：LLM 即你自身（框架出 prompt，你回答）
PYTHONPATH=src python3 scripts/demo_farmer_walkthrough.py

# 对已积累的经验做离线归纳（mock 演示）
python3 scripts/or_experience_cli.py induce --mock-demo --json

# 完整归纳演示：3 个异质问题 → 1 条 validated pattern
PYTHONPATH=src python3 scripts/demo_induction_walkthrough.py

# 跑全套测试
cd tests && PYTHONPATH=../src python3 -m unittest discover -p "test_*.py"
```

---

## 它能做什么

### 1. 求解 OR 问题——多求解器并行探索

- **先建模后动手**：Agent 把问题形式化为 `<think>` + `<model>`（GAMS 风格 DSL），`ModelingGate` 在**任何代码运行之前**校验它（格式/结构/语义三层，最多 3 轮修复）。
- **异构并行探索**：校验通过的模型分叉到最多 **7 个求解器分支**——Gurobi、SCIP、HiGHS、COPT、OR-Tools (CP-SAT)、PuLP、Pyomo——每个分支在隔离沙箱中运行，分支内支持顺序修复。
- **gold 门控闭环**：求解结果与 gold 答案（由调用方提供）对比，不匹配就触发**反思式重新建模**（外层最多 3 轮），而不是盲目重试。

### 2. 积累经验——一个会进化的经验库

- 成功求解与暂存的失败做**成败对比总结**，通过**入库门禁（judge）**才进入经验库。
- **Modeling Bank** 存储建模方法，所有记录同级——直接求解的 (`status=null`) 和归纳产出的 (`status=validated`) 共存。每条记录带 `modeling_aspect`（constraint/objective/variable/classification/structure）和结构签名。Episode 记录问题级快照；派生的修复错误转移图提供错误→修复指引。
- 经验库**实现自进化并保留历史**：记录内容 append-only，但生命周期状态（`active → deprecated`）+ utility 软删除会把坏经验退役进压缩冷库，并防"复活"式重复入库。

### 3. 离线归纳一般原则——"举一反三"

定期（手动或由触发策略门控）从已积累的经验中挖掘**结构同构但业务异质的经验簇**（例如库存/生产/排班三条经验共享同一通用的建模方法或数学技巧），对齐它们的结构角色，归纳出候选原则——**这条原则只有在通过 solver 反例证伪、并在未见任务上证明有迁移改进后，才成为 `validated` pattern**。验证通过的 pattern 随后回流到在线求解，作为建模阶段的有效经验。

完整归纳循环见 [references/induction-pipeline.md](references/induction-pipeline.md)。

### 4. Pattern 回流在线求解

验证通过的记录回流到在线循环，作为**建模先验**。框架在建模前召回相关记录，注入到 prompt 中（`[E1]...`、`[E2]...`），agent 用 `[uses En]` 引用实际使用的经验。gold 匹配后，框架为被引用的记录记 utility——与软删除评分和归纳触发器闭环。这是完整闭环的最后一段：**求解 → 经验 → 归纳 → 回流 → 更好求解**。

---

## 整体结构

```
自然语言 OR 问题
      │
      ▼
结构化建模（think→model→verify）  ◀─── planning priors（[uses En] 引用）
      │
      ▼
7 个求解器并行分支 ──▶ 跨求解器校验 ──▶ 与 gold 匹配？
      │                                    ├─ 否 → 反思式重新建模 ↺
      │                                    └─ 是 → 成败对比总结
      ▼                                              │（入库门禁）
经验库                                             ▼
  Modeling Bank：所有记录同级（status=null | validated）
  Episodes · Implementation · Repair · Solving
  生命周期：active → deprecated → 压缩冷库
      │
      ▼（离线触发）
结构归纳：候选簇 → 编码 → 对齐 → 归纳（假设）
        → 反例证伪（solver）→ 迁移验证（未见任务）
        → 通过 → pattern 入库（append-only）
```

---

## Harness 集成

本框架是 harness agent 下的**工具**。职责划分：

- **框架管理规则**：schema、受控词表、`ModelingGate` 校验器、解析、去重、append-only 存储、**检索**（query 构造、embedding 搜索、排序、过滤）以及所有 prompt 模板。
- **Agent提供LLM**：agent 生成 `<think>`/`<model>`、结构签名、求解器代码、成败对比总结、入库判定，以及（离线）对齐/假设/反例输出。框架从不自行调用 LLM；每个 LLM 触点都走注入的 `llm_client`。

## CLI 命令参考（仅独立模式/演示）

CLI 用于独立模式（无 harness agent：cron、批处理）和演示/测试。harness 模式下请直接使用 Python API。

```bash
python3 scripts/or_experience_cli.py solve       --problem "..." [--solvers a,b,c] [--mock-demo] [--json]
python3 scripts/or_experience_cli.py solve       --interactive-llm --problem-file problem.txt   # harness：你在 stdin 回答
python3 scripts/or_experience_cli.py retrieve    --layer modeling --query "..." [--json]
python3 scripts/or_experience_cli.py append      --input experience.json [--json]
python3 scripts/or_experience_cli.py induce      [--auto] [--min-new-realizations 3] [--mock-demo] [--json]
python3 scripts/or_experience_cli.py stats|rebuild-index|validate-bank [--json]
```

- CLI `solve` 走单步流程（求解+自动提取）。**两步式 harness 流程**（`solve(defer_extraction=True)` + `evaluate_with_gold(gold)`）仅 Python API 可用。
- CLI `induce --interactive-llm` 无法完成迁移验证（transfer solver 会抛 `RuntimeError`）。需走 Python API 注入真实 transfer solver。
- 独立模式需要 LLM 包装命令（`--llm-command`）。
- 运行数据默认在 `~/.hermes/or-experience-bank`，可用 `OR_EXPERIENCE_BANK_HOME` 覆盖。
- `--mock-demo` 用假 LLM + mock 求解器完全离线运行，仅供演示/测试。

## 项目结构

```
src/or_experience_bank/
├── core/          # schema、append-only 存储、生命周期、utility 统计
├── modeling/      # 建模门禁（GAMS 风格 DSL）、签名抽取
├── experience/    # 成败对比提取、入库门禁、失败暂存
├── retrieval/     # embedding 索引、检索器、修复图、modeling 检索器
├── solving/       # 编排器、执行沙箱、反思
├── solvers/       # 7 个求解器适配器 + 注册表
└── induction/     # 候选簇、编码、对齐、归纳、反例、验证、触发、编排
scripts/           # CLI + 演示脚本（farmer walkthrough、induction walkthrough）
references/        # DSL、签名、schema、流程、prompt 模板（随 harness 部署）
tests/             # 237 个 unittest 用例
```

## 文档

| 文档 | 内容 |
|---|---|
| [SKILL.md](SKILL.md) | harness agent 操作契约（你即 LLM） |
| [references/modeling-contract.md](references/modeling-contract.md) | GAMS 风格 DSL 语法、约束标签规则、三层校验 |
| [references/structural-signature.md](references/structural-signature.md) | 签名 schema、受控词表、对齐规则、示例 |
| [references/induction-pipeline.md](references/induction-pipeline.md) | 离线归纳循环详解 |
| [references/experience-schema.md](references/experience-schema.md) | 记录 schema：签名、建模经验、Episode、生命周期 |
| [references/workflow.md](references/workflow.md) | 在线求解流程 |
| [references/prompts.md](references/prompts.md) | 12 种 prompt 模板及完整 input→output 示例 |
| [references/trajectory-schema.md](references/trajectory-schema.md) | AttemptRecord、BranchResult、SolveResult、终止值 |
| [references/solver-adapters.md](references/solver-adapters.md) | 7 个求解器适配器、结果契约、执行控制 |
| [docs/project-overview.md](docs/project-overview.md) | 完整项目介绍（组会汇报用） |
