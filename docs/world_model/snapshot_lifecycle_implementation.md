# 快照职责解耦：实现与验收记录

日期：2026-10-15。对应 [本地审计计划](snapshot_lifecycle_audit_plan.md)；操作协议见 [生命周期、兼容与维护](../../references/snapshot_lifecycle.md)。

## 基线、交付位置与范围

- 原工作目录：`/data/home/zhanyiqi/or-experience-bank`。初始审计时其 `.git/` 为空，无法核实原分支、HEAD 或既有未提交差异；10 月 15 日实施验收阶段未连接或推送远端。
- 审计基线是本地源文件快照，49 个 Python 文件指纹为 `760b2654e43c0dd44326fe5ac4e34c93ba5117ec9487687b91c83975f04a0ed9`。它包含 r24 归档优化，不能冒称恰好等于用户指定的 r23 提交。
- 为形成可审查的隔离差异，在 `/tmp/or-harness-snapshot-review-waf29_j0` 建立独立本地仓库：`local-audit-baseline` 的导入提交是 `50646c8`，交付分支为 `snapshot-lifecycle`。此仓库没有远端，导入提交不是原远端历史。最终 HEAD 由该目录的 `git rev-parse HEAD` 给出。
- 实现同步保留在原工作目录；修改前源码备份为 `/tmp/or-harness-before-implementation.tar.gz`。未修改、迁移或清理用户主库和 DS 库，没有重跑建库。
- 交付源码共 52 个 Python 文件，同审计算法的 SHA-256 为 `275d53f91eb453506bb3e88de2756a101d323f5ff303fb9271175b71ae5ca293`。
- 未更改 strategy_id、结构分组/分箱、收益/代价/风险定义、策略选择公式、归纳门槛、基模或观测版本。`why` 参与策略身份仍是独立问题。

## 完成的职责变化

完整 P/X/H/B 视图在内存中组装，候选共享一份。新流程不向 `belief_snapshots` 写入完整快照，也不再写整批 `knowledge_version` 正文。新上下文格式是 `wm-context/4`，序列化使用轻量 `conditions`，X/B 只存在于 `solving_context`，不嵌套重复完整 snapshot。

`frozen_conditions` 保存实际后续需要的条件事实：问题/进度版本引用、预算聚合与未知原因、真实/假想来源和知识适用范围修订号。它沿用原 pre/post/input ID 关联，不保存覆盖统计、整批知识正文、支持执行清单或成本事件历史副本。问题和进度值、实际引用的证据/知识/反馈/提示内容在 `condition_values` 按内容共享。

`prediction_contexts` 保存轻量审计清单；可复用的完整定向输入存于 `runtime_inputs`。默认有效期 24 小时、最多 128 份活跃输入、活跃 JSON 负载上限 16 MiB。大题面共享其版本，不能据此承诺任意题面都只有几 KB。选择/拒绝消费或成功关闭 episode 后释放完整工作输入；过期清理在后续上下文写入触发，无后台定时器。

模型调用使用独立短事务的租约保护，网络调用期间不持有数据库写锁。调用前保存 `call_state=in_flight`、结果及 usage 未知的预测记录，返回后更新同一预测编号；进程中断留下可检查的未知状态。失联调用的租约默认一小时后可过期，不无限期钉住完整输入。

执行仍是一条真实 attempt 一条事实。保留失败、超时、暂存、未绑定、usage 补录及成本 mask；实际方法未知时不复制计划。采用声明和案例复用引用保存执行前观察到的内容版本，显式 `[]` 仍与未声明不同。脚本、运行后观察到的结果、输出剪辑和内部 CIR 可用不可变产物引用保存；产物写入失败不会改写已观察到的执行成功和成本。

H+ 通过 `knowledge_availability` 的适用范围/存在状态修订和执行条件判断“当时可用”；采用声明另外报告。历史可用性缺失保持未知，不变成“没有采用/没有增益”，也不改变原验证标准。

## 消费者的新来源

| 消费者 | 新来源 | 兼容回退 |
| --- | --- | --- |
| `_episode_progress` | 按 task/episode/problem identity 索引读取条件事实；写 model 不清空进度 | 最新匹配真实旧快照，缺失不臆测 |
| `choose_next`、题面/检查恢复 | 决策条件引用的共享原问题值、执行自身条件和 task_text_digest | 原 root/pre snapshot 或已存题文 |
| 新/旧预测绑定 | `prediction_conditions()` 统一解析冻结来源与问题身份；实际 action 的 pre 条件 | v3 上下文保存值、原 snapshot |
| 预算校准 | 预测自身 frozen budget，或上下文/条件引用；真实预算账本保持原口径 | 旧 snapshot.budget_state |
| 知识目标 | 候选相关统计与冻结 targets、distinct task 计数；不保存全支持 ID 清单 | v3 context / 旧 coverage |
| H+ 延迟验证 | 执行时间、知识适用修订号与 profile；采用声明独立报告 | 原知识可用列表，显式维护后为共享历史 ID 列表 |
| context 读取 | 有效期内读取原 runtime input；过期报 ContextExpired | v3 原格式读取，不读今天银行回填 |
| inspect / context audit | 轻量条件及版本清单，实际请求摘要/遗漏、提示版本 | 旧 payload 保持可读；不支持版本显式报错 |
| 归档 / 清理 | 继续使用原归档与 r24 任务级索引；新维护入口检查依赖 | 未解析的旧引用和归档共享版本默认保护 |

业务消费者不再直接 SQL 读取完整快照。兼容 SQL 仅保留在统一读取、枚举及显式维护入口。

## 兼容与未知边界

旧库可以直接打开，原执行 ID、结果、代价、mask、策略知识和已存校验不重写。`wm-context/3` 有显式读取适配；未知版本不静默略过。正常新任务不会回落到旧完整写入模式。

不能恢复从未保存的旧实际方法、题文、知识正文版本或外部数据历史内容。原来只有条目编号或可覆盖路径的引用仍不能声称具有不可变历史语义；不会拿今天的正文或预算补齐。已保存的真实/假想来源缺失时，新策略预测绑定报告未知并阻止把该预测配成真实校准样本。旧导出缺少工作时间时使用 `created_at_unknown` 显式标记，不默认填今天；H+ 仅在原执行时间或真实 action 结束时间可确认时使用该样本。

已知“未声明预算”与“预算历史缺失”现在明确区分：预测后新增预算不再偷偷成为原预测条件。预算历史确实缺失时，概率评分保持未知，实际成本及测量 mask 保留。相应预算通道测试改为先声明、再预测，执行后补录实际消耗；另有回归证明新预算不会重标旧预测。这属于必要条件可比较性的修复，没有改风险事件、观测单位、测量版本或实际成本。

只有显式旧库维护会将可恢复的大对象转换为条件事实并令相应运行输入过期。默认打开库、预测和 episode close 不重写历史快照 payload。

## 模型输入变化单独列账

- A 存储解耦：固定候选、固定实际命中、固定来源下，对比旧代码捕获的 provider 夹具。候选统计、知识目标、召回身份/正文和关键 provider 输入一致；只规范化编号/时间、声明的上下文格式版本，并将下列 B 投影单独验证。
- B 校准诊断投影：完整 `validity_corrections` 不再进入上下文/模型，改为数量、种类及字段计数；stored/derived 明细和发布摘要独立保留。小夹具可能因计数元数据增加而稍大，不能承诺请求总是缩小。
- 旧 `predict_outcome` 写路径也改为候选相关统计/知识，去掉增长的支持 ID 清单；这是旧 provider 输入变化，不能算纯存储等价。旧记录的读协议保留。
- 请求保留最终实际展示引用、内容版本、遗漏、字符大小和摘要；字符估算不是 tokenizer 实测。没有真实模型 token、时延或质量改进结论，也不承诺随机模型逐字一致。

## 同一夹具的实测

固定一个候选和一个实际命中的知识条目，加入同格无关策略；不调用模型或求解器。总量包含快照/上下文/轻量条件/共享值/适用索引/旧知识批次的 JSON payload，不含两边不变的真实执行事实。详见 [机器可读数据](snapshot_lifecycle_measurements.json) 和 [复现脚本](repro_snapshot_lifecycle.py)。

| 同格策略数 | 原初始 payload 字节 | 新初始 payload 字节，含活跃输入 | 原 5 动作增量 | 新 5 动作增量 | 新输入消费后总量 | 原/新完整请求字符估算 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23,862 | 27,451 | 43,024 | 13,763 | 25,030 | 39,381 / 39,563 |
| 10 | 35,548 | 27,456 | 72,234 | 13,769 | 25,037 | 39,383 / 39,565 |
| 100 | 152,373 | 27,457 | 364,285 | 13,775 | 25,043 | 39,386 / 39,565 |

原实现 5 个动作后完整快照 6 行；新实现始终为 0 行，保留的是 6 条必要轻量条件事实。一次比较只有一个活跃上下文；消费后 `runtime_inputs` 为 0 行，审计清单仍在。新负载没有随无关策略线性增长。上述数值有少量编号/时间字符波动。

没有对这个对比夹具执行 VACUUM。没有真实 1.30 GB 库副本，因此不报告真实库已缩小或缩小百分比。维护测试在临时库上验证了 preview、备份、转换、依赖保护、幂等与恢复，不能替代用户主库预览。

## 准确操作入口

正常 `plan-next → choose-next → execute → record/check/close` 保持原命令；无需额外调用 `snapshot`。输入已消费不妨碍执行/绑定已发预测。

```bash
# 已消费/过期输入的审计清单
orx --home /work/run context --context-id ctx_ID --audit

# 固定记忆测试：每题从同一完成建库后的基库复制
orx --home /work/fixed-base fork-memory --destination /work/test-A
orx --home /work/fixed-base fork-memory --destination /work/test-B
# A 重试继续 test-A；B 的全部操作使用 test-B，结果不回灌基库。

# 默认只读预览
orx --home /work/bank-copy maintain-snapshots
# 显式应用到副本；备份必须为新文件
orx --home /work/bank-copy maintain-snapshots --apply --backup /work/backups/bank-before.db
# 可选物理回收需要另行显式请求 --vacuum。
# 需要过期旧的长期未关闭视图时，先预览 --expire-open-days 30。
```

`fork-memory` 用 SQLite backup 包含 WAL 数据，并复制固定 index/archive/artifacts。基库需要停止写入和索引更新；每题目录不得已有内容。题内可记录真实尝试与检查、更新预算与进度。仅禁止归纳不足以冻结跨题记忆。

清理在写锁内备份、校验、重新检查依赖并原子提交。显式过期旧视图不删除 pending 执行或结束 action，不把 H+ 标为已验证。未解析引用仍受保护；归档引用未解析时保留共享旧版本。新共享值/产物文件暂不做垃圾回收，防止跨引用误删。

## 文件与验证

主要业务文件：`api.py`、`cli.py`、`core/schema.py`、`core/storage.py`、新增 `core/artifacts.py`、`execution/executor.py`、`strategy/strategic_bank.py`；世界模型的 `conditions.py`（新增）、`snapshot_maintenance.py`（新增）、`context.py`、`strategy_prediction.py`、`service.py`、`episode_closeout.py`、`maintenance_decision.py`、`attribution.py`。

测试新增 `test_snapshot_lifecycle.py`、`test_snapshot_equivalence.py` 和旧 provider 输入夹具；调整原快照/XB 实现细节断言、消费后读取、校准诊断投影与预算声明时点。继续运行 r22/r24 性能约束、r23 预算标签、H+、方法/关联/校准/窗口/CLI 等回归。

文档更新中英文 README、主 SKILL 的职责/入口段、相关 references，以及预测上下文、episode closeout、能力演化示例和文档检查器。主 SKILL 没有扩成协议长文。

核验命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests/harness -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python references/examples/_check_docs.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python docs/world_model/repro_snapshot_lifecycle.py
```

最终全量 **1,710 项通过，耗时 121.342 秒**，日志为 `/tmp/or-harness-snapshot-acceptance.log`。定向预算/校准兼容 124 项、H+/trace 43 项、r22/r24/等价性 20 项以及跨 CLI 三进程单项通过。文档检查通过；13 个文档列出的示例已逐个实际运行通过，不把“脚本存在检查”当成执行通过。

首轮沙箱禁止创建本地 HTTP stub，相关测试在获准的沙箱外执行；一次批量示例的自动审批超时，改为单独运行仓库中的两个 HTTP 示例后通过。期间发现的 JSON 层级测试错误、预算夹具时点和示例旧断言已修正，没有通过放宽业务断言掩盖失败。

回退方式：原源码快照和本地审查基线提交均保留。真实库未执行维护，因此没有真实库回退动作。若用户以后显式应用维护，备份应恢复到独立 home 并保留匹配产物/索引；已经产生新格式事实后，不应直接用旧代码忽略它们。

## 完整修改文件清单

下列 41 个文件相对于本地审计导入提交发生变化：

```text
README.md
README_zh.md
SKILL.md
docs/world_model/repro_snapshot_lifecycle.py
docs/world_model/snapshot_lifecycle_audit_plan.md
docs/world_model/snapshot_lifecycle_implementation.md
docs/world_model/snapshot_lifecycle_measurements.json
references/commands.md
references/episode_closeout.md
references/examples/_check_docs.py
references/examples/capability_evolution.py
references/examples/episode_closeout.py
references/examples/prediction_context.py
references/prediction_context.md
references/snapshot_lifecycle.md
references/strategy_outcome.md
references/world_model_contract.md
src/or_harness/api.py
src/or_harness/cli.py
src/or_harness/core/artifacts.py
src/or_harness/core/schema.py
src/or_harness/core/storage.py
src/or_harness/execution/executor.py
src/or_harness/strategy/strategic_bank.py
src/or_harness/world_model/attribution.py
src/or_harness/world_model/conditions.py
src/or_harness/world_model/context.py
src/or_harness/world_model/episode_closeout.py
src/or_harness/world_model/maintenance_decision.py
src/or_harness/world_model/service.py
src/or_harness/world_model/snapshot_maintenance.py
src/or_harness/world_model/strategy_prediction.py
tests/harness/fixtures/snapshot_input_v3.json
tests/harness/test_calibration_v2.py
tests/harness/test_episode_closeout.py
tests/harness/test_prediction_context.py
tests/harness/test_r23_budget_labels.py
tests/harness/test_snapshot_equivalence.py
tests/harness/test_snapshot_lifecycle.py
tests/harness/test_task_check.py
tests/harness/test_world_model.py
```

## 2026-10-23 上传交付

经用户授权，重新克隆并核对远端 `world-model-integration`，实际基线仍是 `f67e9a85f88e059344cde5bbb0f60a5ca0acecd9`（r23）。交付分支为 [`snapshot-lifecycle-20261023`](https://github.com/jackyz17/OR-Harness/tree/snapshot-lifecycle-20261023)，以该真实基线为父提交，提交历史不包含独立本地仓库的导入根。

本地基线比远端多出的 r24 业务差异仅为归档/证据窗口使用任务级 execution→action 索引的性能修复。先单独保存该前置修复（`episode_closeout.py`、对应参考文档、`test_r24_archive_scan.py`），再应用本轮快照改造。相对远端共 42 个修改文件；上文 41 个文件的清单仍针对原本地审计导入基线，新增的第 42 个文件是 `tests/harness/test_r24_archive_scan.py`。

远端原有 `.gitignore`、`assets/` 和 `harness_review_bench/` 保留。仅显式纳入本轮的 4 个交付文档/复现文件；不上传本地其他历史笔记或数据库。上传工作副本为 `/tmp/or-harness-upload-20261023`，其中源码、测试、参考文档逐文件比对与原验收版本一致，源码指纹未变化。

在真实远端历史的移植副本上重新运行全量回归：**1,710 项通过，122.799 秒**（`/tmp/or-harness-upload-tests.log`）；文档检查通过（`/tmp/or-harness-upload-docs.log`），`git diff --check` 通过。前次实际运行的 13 个示例结果保留为历史验收，没有将本次文档存在性检查重复计作运行。

PR 以 `world-model-integration` 为目标，上传操作不自动合并 PR。用户主库和 DS 库仍未修改。
