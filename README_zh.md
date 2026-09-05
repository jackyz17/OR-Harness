# OR Experience Bank（中文版）

一个面向 LLM Harness Agent 的 Python 框架，实现**agent 自主选择求解器求解 OR 问题**、**积累经验证的求解经验**、以及离线地**从历史经验中归纳出新的优化原则**（"举一反三"）。纯 Python stdlib，零依赖。

[English](README.md)

---

## 快速上手

```bash
# 环境自检（新环境的第一步）
python3 scripts/orx.py doctor

# 初始化经验库
python3 scripts/orx.py init

# 跨进程续跑演示（两个会话、一个 run——无 server、内存零状态）
python3 scripts/demo_orx_resumability.py

# 跑全套测试
cd tests && PYTHONPATH=../src python3 -m unittest discover -p "test_*.py"
```

---

## 架构：LLM-as-orchestrator + framework-as-tools

Harness agent（Claude Code / openclaw / Hermes 类）是编排者：读 [SKILL.md](SKILL.md)，思考，每步调用一条**无状态 `orx` 命令**。所有跨调用状态都在**文件**里（run 目录），从不在 server 进程里——每条命令都是独立进程，run 天然跨越连接断开、会话切换、进程重启。

**链路由盖章（stamp）强制，而非 token。** 每个门禁命令给它批准的产物盖一个内容 hash 章；下一条命令发现前置章缺失或文件在盖章后被改动就拒绝执行。步骤不可跳过，但每一步都可以自由重试。

```
自然语言问题
  │
  ▼
orx recall ──► priors.json（[E1],[E2],... 回流）
  │
  ▼
agent 写 model.txt ──► orx validate（L1+L2 门禁）──► 盖章
agent 写 signature.json ──► orx signature（词表门禁）──► 盖章
  │
orx hints ──► 写码前先拉银行提示
agent 写 branches/<solver>/solve.py ──► orx solve（沙箱执行，单一求解器）
  │
orx gold ──► 与用户提供的 gold 匹配？
  ├── 是 ──► orx append ×N（每层经验一条）──► orx episode（终态）
  └── 否 ──► 反思 ──► orx new-round ──► 重新建模（≤3 轮）
```

## 它能做什么

### 1. 求解 OR 问题——agent 自主选择求解器

- **先建模后动手**：Agent 把问题形式化为 `imd` + `<model>`（GAMS 风格 DSL），`orx validate` 在**任何代码运行之前**校验它。
- **agent 自主选择求解器**：每个 run 由 agent 根据可用性、问题匹配度和银行积累的 hints 选择**一个**求解器——Gurobi、SCIP、HiGHS、COPT、OR-Tools (CP-SAT)、PuLP、Pyomo。分支失败只需改该分支的代码重跑 `orx solve`，**链路从不重启**。
- **gold 门控闭环**：求解结果与 gold 答案（**只**来自用户/题目）对比，不匹配触发**反思式重新建模**（`orx new-round`，外层最多 3 轮），而不是盲目重试。

### 2. 积累经验——一个会进化的经验库

- 经验按层合成（modeling / implementation / repair / solving），经 `orx append` 入库，带内容 hash 去重和防复活检查。
- **Modeling Bank** 存储建模方法，所有记录同级——直接求解的 (`status=null`) 和归纳产出的 (`status=validated`) 共存。Episode 记录问题级快照；派生的修复错误转移图提供错误→修复指引。
- 经验库**实现自进化并保留历史**：记录内容 append-only，但生命周期状态（`active → deprecated`）+ utility 软删除会把坏经验退役进压缩冷库，并防"复活"式重复入库。

### 3. 离线归纳一般原则——"举一反三"

定期（由触发策略门控）从已积累的经验中挖掘**结构同构但业务异质的经验簇**（例如库存/生产/排班三条经验共享同一通用的建模方法或数学技巧），对齐它们的结构角色，归纳出候选原则——**这条原则只有在通过 solver 反例证伪、并在未见任务上证明有迁移改进后，才成为 `validated` pattern**。验证通过的 pattern 随后回流到在线求解，作为建模阶段的有效经验。

完整归纳循环见 [references/induction-pipeline.md](references/induction-pipeline.md)。

### 4. Pattern 回流在线求解

验证通过的记录回流到在线循环，作为**建模先验**。`orx recall` 在建模前召回相关记录；agent 用 `[uses En]` 引用实际使用的经验。gold 匹配后，`orx episode` 为被引用的记录记 utility——闭环完成：**求解 → 经验 → 归纳 → 回流 → 更好求解**。

---

## Harness 集成

以 **skill** 形式分三层部署（完整指南见 [docs/deployment.md](docs/deployment.md)）：

```bash
# 第一层：框架（每台机器一次）——把 `orx` 装上 PATH
pip install -e ".[solvers-free]"

# 第二层：bank（每台机器/团队一次）
orx init

# 第三层：skill 文档（每个 harness 一次）
cp -r <repo> ~/.claude/skills/or-experience-bank/     # Claude Code 示例
```

之后 agent 用原生 shell 工具驱动一切。无 server 进程、无连接管理、无 MCP 配置。新环境中 agent 的第一个动作是 `orx doctor`（环境自检）；中断后用 `orx status` 重新定位。

## orx 命令参考

```bash
# 部署
orx doctor                        # python / bank / 求解器 / 索引自检
orx init                          # 初始化 bank 目录

# 在线求解链（run 目录 = 当前目录）
orx recall --problem-file p.txt   # 开始 run + 召回建模先验
orx validate                      # model.txt 的 L1+L2 门禁 → 盖章
orx signature                     # signature.json 的词表门禁 → 盖章
orx hints --solver <s>            # 写码前拉银行提示
orx solve --solver <s>            # 沙箱执行所选求解器（修复重试）
orx gold --answer <v>            # 与用户提供的 gold 对比
orx gold [--answer <v>]           # 记录 gold 判定（用户提供 / 仅一致性）
orx append --file exp.json        # 入库一条经验（gold 门禁强制）
orx episode                       # 终态：episode + utility 归因
orx new-round                     # 归档当前轮产物，进入反思轮
orx status                        # 我在哪、下一步是什么

# 银行
orx query --layer <L> --query "..."   # 检索 modeling/implementation/repair/solving
orx show --id <id>                    # 取一条完整记录
orx deprecate --id <id> --reason "..."# 退役一条记录（进冷库）
orx stats                             # 银行统计

# 离线归纳（每簇一个目录，位于 <bank>/induction/<cluster_id>/）
orx trigger / clusters
orx align / induce / refute / validate-pattern / append-pattern --cluster <id>
```

每条命令在 stdout 输出单个紧凑 JSON 对象（ReAct 的 Observation）；长内容写入 run 目录的文件。

## 项目结构

```
src/or_experience_bank/
├── core/          # schema、append-only 存储、生命周期、utility 统计
├── modeling/      # 建模门禁（GAMS 风格 DSL）、签名抽取
├── experience/    # 成败对比提取、入库门禁、失败暂存
├── retrieval/     # embedding 索引、检索器、修复图、modeling 检索器
├── solving/       # 编排器、执行沙箱、反思
├── solvers/       # 7 个求解器适配器 + 注册表
├── induction/     # 候选簇、编码、对齐、归纳、反例、验证、触发、编排
└── cli/           # orx：基于文件 run 的无状态命令（盖章链）
scripts/           # orx 入口 + 演示脚本
references/        # DSL、签名、schema、流程、例子、生命周期、求解器
tests/             # unittest 测试
```

## 文档

| 文档 | 内容 |
|---|---|
| [SKILL.md](SKILL.md) | agent 操作合同（基于 `orx` 的 ReAct 工作流） |
| [references/workflow.md](references/workflow.md) | 命令调用视角的端到端流程 |
| [references/modeling-contract.md](references/modeling-contract.md) | GAMS 风格 DSL 语法、AUXILIARY 块、L1/L2/L3 校验 |
| [references/structural-signature.md](references/structural-signature.md) | 签名 schema、受控词表、对齐规则、示例 |
| [references/experience-schema.md](references/experience-schema.md) | 记录 schema：ModelingExperience、ExperienceRecord、Episode、生命周期 |
| [references/bank-lifecycle.md](references/bank-lifecycle.md) | utility 归因、软删除、冷库、防复活 |
| [references/induction-pipeline.md](references/induction-pipeline.md) | 离线归纳循环 + orx 命令映射 |
| [references/examples.md](references/examples.md) | 3 个完整示例：正例 / 歧义 / 反例 |
| [references/solver-adapters.md](references/solver-adapters.md) | 7 个求解器适配器、result.json 契约、沙箱规则、求解器 API 笔记 |
| [docs/deployment.md](docs/deployment.md) | 部署指南：框架 / bank / skill 三层 |
| [docs/project-overview.md](docs/project-overview.md) | 项目完整介绍（组会汇报材料） |
