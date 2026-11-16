# 6 P1 + 3 P2: evidence, versions and induction transactions

本轮基线为 `0bcedd44e8bb4425e5d9932c7d4705ffcd25e65f`。改动在服务器当前目录实现并运行测试，保留原有修改和 12 份本地专有文档。当前目录没有可用的 `.git` 元数据：`git status --short`、`git rev-parse --short HEAD` 以及 `git diff --check` 均无法按工作树方式运行。本轮通过交付清单逐文件哈希确认起点，并核对 GitHub `main` 为上述提交；远程提交号没有冒充本地 HEAD。修改前副本位于 `/tmp/or-harness-nine-before`。

冻结 H+、agent 决定归纳/发布/效用、冷启动推导与构造例的范围保持原义。G/C/R 预测、效用公式和求解门控没有改动。

## 逐项复现与验收

| 项目 | 复现与修复 | 正常入口验收 |
|---|---|---|
| P1-1 回执依据留存 | A 触发；B 为无预测、无归纳待办且未出现在材料页或规则种子中的历史执行，只由比较引用。提交前解析两类 comparison 的直接依据，保存最小事实、checker/artifact 引用，并通过回执和规则版本事件关联；完成后释放临时完整副本。 | 规则比较和顶层 review 分别引用两个历史执行，标签为 1；正常窗口清理、重启后仍可读原依据与标签，下一次 provider 请求及 `induction-status` 正常。排除关键依据后新标签未知。已丢失依据的旧回执单独未知，其余反馈继续可用。新不存在引用拒绝并保留原待办。 |
| P1-2 可选成本对照 | 无预测的真实基线在登记对照后、当前任务收尾前被自然淘汰，旧版 close 抛错。正常采用结算改为返回 `cost_comparison.state:unavailable`、基线 ID 和具体原因。 | H+=1 与 0 均保留原题 `passed` 和真实使用结算；前者创建唯一待办并可空审查，重试同一待办，后者无新归纳。另测已排除、不存在、引用损坏的基线。显式独立 `validate_case` 输入错误仍给出清楚的校验信息。可读对照仍逐维返回双方实测和来源。 |
| P1-3 具体有效依据 | 旧版缺失 artifact 路径可借其他通过的检查形成标签；原通过结果改为失败或缩减覆盖后，标签仍不变。现在校验所属执行上的真实路径、检查项与捕获文件，并冻结支持所依赖的 checker 结果和覆盖。 | 不存在的 artifact、越界检查项和不存在字段均拒绝；唯一通过依据经正常 `check-task` 更正为失败或去掉原覆盖后，后续反馈未知，原时点和已发预测输入仍为原标签。相同重复检查、迟到首次检查及其后续更正、显式旧 checker 引用、真实失败支持边界均覆盖。只引用某个检查项时，不借用也不依赖无关检查项。 |
| P1-4 选定版本管理 | v1 已发布时，合并到候验 v2 会提前移除来源；辅助执行 v2 可覆盖主执行实际采用的 v1。现在合并落实同时检查目标具体版本和来源冻结版本，归因以主执行冻结采用版本为准。 | A v1 在用，A+B 候验 v2 时 B 仍可用；显式发布 v2 后 B 退出向量索引与日常检索，旧使用仍可读，迟到 promote 被拒绝且不增加效用事件。主执行 v1、辅助 v2、声明 demote v2 在写入前拒绝，v2 保持 active。既有 H+=0 promote/demote/keep 回归通过。 |
| P1-5 冻结检查 | 维护分流绕过正式归纳授权时，也绕过冻结。现在 `induce` 及维护事务内复用模式检查。 | 冻结包含候验 v2 与已发布 v1 的库：API 发布候验以及真实 CLI `--attribute-effect` 暂停请求均拒绝，版本、效用事件及向量内容不变。正常建库模式中同类已有条目维护仍成功，无需正信号待办。求解、使用记录和原题检查仍走原入口。 |
| P1-6 提交后索引 | 旧版先更新 sidecar 向量，再因回执坏引用回滚 SQLite，使向量与规则正文分离。现在事务内保存规则、动作、校验、回执和待同步条目，提交后才写向量。 | 先发布并真实向量召回 v1；v2 因不存在依据被拒绝后，重启仍仅有 v1 和原向量，待办可重试。有效提交 v2 时模拟一次外部同步异常，规则/回执已提交，返回 deferred；相同请求重试恢复向量召回 v2，动作和回执各一份。断言外部同步不在 SQLite 事务内。 |
| P2-1 补证据归属 | 顶层未传 ID/receipt 的纯 `add_evidence` 被提前分流维护，忽略关系内 `material_id`。现在显式待办和材料归属优先于维护分类。 | 真实 CLI 只传 `--relation`，含已有条目、add_evidence 和本次 material_id，成功结清原待办；等价 API 重试返回相同回执。跨待办引用先拒绝，无关联的补证据仍为 maintenance-only。旧提交默认参数读法保持兼容。 |
| P2-2 预算分页 | 长旧规则先占预算，下一对照交付失败，游标反复停在同一页。现在在旧规则和可选邻例之前保护下一对照，必要时交付真实详情引用；最后裁剪保留实际已交付的对照和游标。 | 7 条较长旧规则，较早 A 触发、较新 B/C 同题执行，每页一条对照；32,000、2,400、1,500 和 1,024 字符预算均读到 B/C 或有效引用，游标推进且最终结束。API 结果及 CLI envelope 的实际序列化计数均不超预算。 |
| P2-3 升级 checker | 旧 `rule_evidence_latest` 完整副本优先于新事实，淘汰后又显示未检查。现在从在线记录或各保留来源中选择最新 checker；对已引用的旧副本检查更正时保存新最小事实，历史副本继续可查。 | 合成旧格式回归覆盖重启和历史读取；另用实际保留的 `40581f7` 源码建未检查旧库，再用新代码正常补 check、完成有依据无增量审查、窗口淘汰并重启：checker 仍为新 passed，标签仍 0，检查前时点为未知。 |

复现日志保留在 `/tmp/review-nine-before.log`、`/tmp/review-nine-before-confirmed.log` 和 `/tmp/review-nine-pagination-before.log`。首次补测时纠正了 fixture 的索引属性名及 CLI/API 默认参数差异；之后在未修复源码上重跑关键用例，确认弱点确实来自基线实现。新增 17 个测试方法，扩展原有 fixture，覆盖正常 CLI/API 入口。

## 改动文件

- `src/or_harness/strategy/evidence_access.py`：最小事实重读、旧完整副本兼容、最新 checker 选择和已引用依据刷新。
- `src/or_harness/world_model/hplus_support.py`：具体引用与 artifact 校验、支持检查快照、迟到/更正/历史观察、单条旧坏依据降为未知。
- `src/or_harness/world_model/hplus.py`：完成前留存全部直接依据，回执/版本关联，正式材料归属和等价提交重试。
- `src/or_harness/strategy/rule_support.py`：可选成本基线缺失/不可读的逐项状态；保留双方实测事实。
- `src/or_harness/strategy/rule_versions.py`：精确版本合并、可选采用结算、主执行归因防护、提交内待同步意图。
- `src/or_harness/strategy/strategic_bank.py`：已确认合并来源不能经生命周期管理恢复。
- `src/or_harness/strategy/index_sync.py`：合并来源的索引移除、重建及健康统计一致。
- `src/or_harness/world_model/induction_material.py`：对照优先与最小预算下的可前进分页。
- `src/or_harness/api.py`：正式引用优先、冻结维护检查、提交后索引同步与幂等修复、主执行采用版本校验。
- `tests/harness/test_review_eight.py`：17 项新增回归及预算、CLI、重启、真实向量召回和同步异常覆盖。
- `SKILL.md`、`references/hplus.md`、`references/induction.md`、`references/commands.md`：同步操作与故障恢复规则。
- 本报告、`run_review_nine_upgrade.py`、`review_nine_demo.json`、`review_nine_upgrade.json`：可重跑升级入口及实测结果。

## 验证命令与结果

```bash
PYTHONPATH=src:tests/harness python -m unittest -q test_review_eight test_rule_versions test_hplus_binary test_episode_closeout test_evidence_window test_induction_material test_r20_induction_chain
PYTHONPATH=src:tests/harness python -m unittest discover -q -s tests/harness
PYTHONPATH=src python docs/validation/run_review_eight_demo.py
PYTHONPATH=src python docs/validation/run_review_nine_upgrade.py --legacy-src /path/to/40581f7/src
```

最终定向 **172 项通过，43.093 秒**；全量 **1,893 项通过，159.827 秒**。日志为 `/tmp/review-nine-directed-complete.log`、`/tmp/review-nine-full-complete.log`。最小预算及相关取材回归额外 **14 项通过，1.373 秒**。相对基线增加 17 项。全量日志含负例 CLI 帮助输出和 HTTP 替身 fixture 的 socket `ResourceWarning`，没有断言失败或错误。

冷启动演示：A=2699、B=90、构造边界=0，在真实 C 之前由 agent 发布；后续真实 C=196，H+=0 keep 不产生新归纳动作，H+ 学习标签为 1。输出与下一次 provider 请求片段见 [review_nine_demo.json](review_nine_demo.json)。原演示和历史报告保持不变。

实际升级演示输出见 [review_nine_upgrade.json](review_nine_upgrade.json)。本次 legacy-src 为 `/tmp/or-harness-agent-review-before/src`；其中 66 个源码文件逐一匹配上一轮保存的 `40581f7` 起点清单。脚本把旧源码作为只读输入，在独立临时库中建库、切换新代码、补 check、结清、淘汰、重启并检查历史；不会改写旧 checkout 或正式记忆库。

原生 `git diff --check` 因无 `.git` 不能运行；已对修改前副本与最终文件运行 `git diff --no-index --check`。该模式状态 1 表示存在差异，需结合实际诊断输出判断空白错误，不能把所有状态 1 当成错误。
本轮 18 个改动文件的该项检查没有空白或冲突标记诊断。

## 操作与范围

正常建库仍为 predict → execute/check/record → close → induction-material → induce。正式补证据可直接传已交付材料引用：

```bash
orx induction-material --induction-id "$J"
# M 使用上一步返回的 material_id；ENTRY、EX 来自实际条目和执行。
orx induce --relation "{\"entry_id\":\"$ENTRY\",\"operation\":\"add_evidence\",\"material_id\":\"$M\",\"evidence\":[\"$EX\"]}"
orx induction-status
```

`committed:true` 且 `index_sync.state:deferred` 表示规则和回执已成功提交。原样重试该提交修复索引，或用现有 `rebuild-index`；不会再次归纳或增加动作/回执。`cost_comparison.state:unavailable` 只指出该可选对照缺失，不取消原题检查和冻结的 H+ 待办。

演示预测器与评审结论采用替身，求解脚本、checker、临时 SQLite 库、CLI 和本地向量索引真实运行。这是机制与恢复链路验收，未据此推断模型判别力或工业收益。真实远程嵌入服务故障、任意进程崩溃时点及长期并发负载未做实盘验证；已覆盖重启、一次提交后同步异常以及既有双连接回归。无法恢复依据的历史回执按要求保持未知，框架没有补造支持。
