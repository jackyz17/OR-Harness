# OR-Harness（中文说明）

一个**小巧、可组合、对 harness 友好**的运筹优化（OR）策略学习能力层。

> 给定一个不断演进的大规模工业优化任务流，OR agent 能否从既往执行中学到"哪些求解策略适合哪些问题结构"，并以更低执行成本持续获得同等或更优的解质量？

OR-Harness 运行在外层 harness agent（Hermes 类）**内部**。它不是自治 agent：没有对话循环、没有运行时 LLM 调用、没有隐藏全局状态。外层 agent 负责编排；本能力层提供建议、执行与记忆。

## 核心能力

- **双层记忆**：append-only 的 **Execution Evidence Bank**（执行证据层：实际策略/实际质量/实际代价/失败与恢复/产物；近期原始 + 代表性原始 + 压缩摘要三类保留）+ 派生的 **Strategic Knowledge Bank**（战略知识层：期望质量/期望代价/期望风险,带预测区间、前瞻校验与归纳时证据验证；可从保留证据重新归纳,入库后有效性不依赖原始证据存活）。条件统计即时计算、永不落库。
- **确定性画像**：从任务 spec 或外层供给的 annotations 提取结构耦合特征——不建 NLP 子系统。
- **外层保有全部控制权的策略召回**：透明评分（`α·Q̂ − β·C_scalar − γ·R̂`）、双层证据回退、四消融模式。冷启动时无先验分数——没有经验就如实返回 `no_memory`，从零积累。
- **沙箱执行**：AST 策略 + POSIX rlimit + 墙钟超时；基本验证；五维代价计量（retries 计入代价）。
- **由你掌控的归纳**：每次 record 后 C1–C6 证据 hint；`induce` 永远是外层显式调用。范围阶梯 L1→L2→L3，加宽可证伪；冷归档防复活。触发准则不再引用先验分数，改为纯统计判据。
- **七个求解器适配器**（highs、pulp、ortools、scip、copt、pyomo、gurobi）——仅做可用性探测，具体求解器由你按情况选择。

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
- **[references/concepts.md](references/concepts.md)** —— 双层记忆、CostVector、处置阶梯（英文）
- **[references/induction.md](references/induction.md)** —— C1–C6、范围阶梯、前瞻校验（英文）
- **[references/examples.md](references/examples.md)** —— 三个完整走查实例（英文）

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
