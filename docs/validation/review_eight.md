# P1-1 至 P1-8 验收（2026-11-13）

本轮基线为 `main@1a836a1a061a154bb2132a84e7ecd0e2b308e914`，父提交
`cdf29f6834806a7adb6c0a38513ba7ae472ee513`。服务器当前目录的 `.git` 不含
可用 Git 元数据，实际运行 `git status --short` 和 `git rev-parse --short HEAD`
均返回 `not a git repository`。通过 GitHub 连接读取 main、提交父链和树，并逐文件
核对本轮开始前保留的本地内容与该基线；未把远程 HEAD 冒称为本地 HEAD。
已有 12 份本地文档保持原样，远程未在工作树出现的文件由原树保留。

## 八项修复及结果

新增回归集中在 `tests/harness/test_review_eight.py`（30 项），共用
`review_fixtures.py`。题目使用普通 task、冻结 raw data/model、真实 Python
整数枚举、执行回执及正常 API；不填母题字段。预测器和人工归纳/核验判断为透明替身。

| 项目 | 实现位置 | 对应回归与实测结果 |
|---|---|---|
| P1-1 原题身份与独立性 | `strategy/evidence_views.py`、`rule_versions.py` | `test_ordinary_A_B_rule_independent_C_196_and_nonmonotone_boundary_0`：A=2699、B=90 共同提炼，独立 C=196 和边界=0 后发布。同一 B 改编号/换文字保留原始数据和模型均不能迁移；单侧 parent 链被合并。`test_strong_new_inputs_override_identical_generic_prose`：不同原始输入的同构题可验证。纯文字具体重复线索保留待确认。 |
| P1-2 新版本回归 | `strategy/rule_support.py`、`rule_versions.py` | `test_v2_real_reruns_resolve_old_cases_and_publish_after_independent_transfer`：同输入/验收的旧重复成功合为 3 个代表案例，用新的执行 ID、真实 v2 采用结清，保留新旧 ID 和 `actual_version_use`。换数据、弱 checker、v1 采用仍有缺口。`test_merge_into_existing_candidate_also_requires_source_cases_on_new_execution_ids`：合并到已存在候验版本时，来源回归也阻止提前发布/合并。历史事实复查另记 `historical_review`。 |
| P1-3 取材及会话归属 | `world_model/hplus.py`、`induction_material.py`、`maintenance.py`、`induction_gate.py`、API/CLI | `test_exact_trigger_and_all_material_sessions_are_owned_by_the_round`：同题 1 后跑 0，主例仍是原 1；改变 purpose/query/count/选择返回新会话，相同查询复用。仅引用历史 B 且省略材料/待办 ID 时，唯一所属轮次自动关联 A 与全部已读会话。ID 冲突在写入前拒绝；另一轮读过的 B 不泄入当前轮。手工 allow 发起可引用 skip 对照；旧 close 重试不覆盖新决定。 |
| P1-4 日常核验及成本对照 | `rule_support.py`、`rule_versions.py`、`api.py` | `test_H0_cost_trial_automatically_finds_measured_nine_to_three_baseline`：H+=0、原待办完成后，正常检查/收尾把同题相同验收、实际条件和计量范围下 9→3 tool_calls 登记为有效对照，保存两边来源。成本数字来自测试宿主计数替身，不用于宣称性能提升。`agent_estimate` 的 100→1 tokens 仍未知；多个不同实测基线给出已有对象，指定后能结清。 |
| P1-5 更正、撤回与历史 | `evidence_access.py`、`rule_support.py`、`rule_versions.py`、`api.py`、`cli.py`、`hplus_support.py` | `test_omitted_optional_review_preserves_support_with_two_withdraw_restore_cycles`：同 checker 省略可选评审保持有效；两轮撤回/恢复得到 active→paused→active→paused→active 的 5 个不同事件，H+ 同步失去/恢复依据。显式空列表或匹配的 `withdrawn` 为撤回；有另一份有效迁移支持时仍可发布。`test_cli_cost_fact_correction_pauses_and_restores_dependent_rule_and_hplus`：正常 amend-cost 将成本对照从 9 改成 1 后撤销“3 更省”的支持，恢复可信 9 后规则及 H+ 恢复；发布变化同步索引。检查链、有效性版本和之前已发出的上下文保留。 |
| P1-6 唯一身份及交错写入 | `core/storage.py`、`world_model/strategy_prediction.py`、`hplus.py`、API/CLI | `test_omitted_close_episode_does_not_mint_a_second_job`：显式 episode 后省略重试仍一份待办。`test_two_connections_late_material_does_not_reopen_completed_job`：两个连接在材料构建/提交之间交错，最终保持 completed，一项真实归纳动作、原回执；同提交缓存返回，冲突拒绝。多次真实正信号分别结清；歧义返回可读选项。`--execution EX` 可选择未命名 episode；精确 NULL 查询不等于查询所有 episode。 |
| P1-7 窗口后的证据访问 | `evidence_access.py`、`induction.py`、`induction_material.py`、API inspect/check/correction | `test_evicted_pending_trigger_and_selected_contrast_survive_restart_and_cli_read`：小窗口真实淘汰触发与已读对照，重开后 CLI 仍读到实际方法/checker，正常取材和提交成功，未重新插入原 bank 行。`test_withdrawn_frozen_source_is_not_resurrected_for_submission`：冻结证据明确撤回后不能支持规则，可空审查完成；恢复也不回填在线窗口。 |
| P1-8 有依据的增量及反馈 | `world_model/hplus_support.py`、`hplus.py`、`rule_support.py` | 格式等价复用原版本；同义改写即使该版本通过发布也不自动成为学习正例。`test_v2_real_reruns_resolve_old_cases_and_publish_after_independent_transfer` 用有来源的 when/do 扩展对照及独立检验得到 1。已有覆盖+纯补证据得到 0；迟到 checker 只进入后续观察。加入 6 个较新未知回执后，相关可靠正反例仍能进入下一 provider。原假设命中和实际知识增量分开记录。 |

## 实际改动文件

- 正常入口及精确查询：`src/or_harness/api.py`、`cli.py`、`core/storage.py`、
  `world_model/strategy_prediction.py`、`world_model/episode_closeout.py`。
- 证据、回归、发布：`strategy/evidence_views.py`、新增 `evidence_access.py`、
  新增 `rule_support.py`、`rule_versions.py`、`induction.py`。
- 调度、取材、反馈：`world_model/hplus.py`、新增 `hplus_support.py`、
  `induction_material.py`、`maintenance.py`、`induction_gate.py`。
- 兼容替身：`experiments/hplus.py` 只给整数演示归纳器补充有来源的机制对照，
  没有改动实验分组、冻结逻辑或漏判统计。
- 测试：新增 `tests/harness/review_fixtures.py`、`test_review_eight.py`；
  更新 `test_hplus_binary.py`、`test_rule_versions.py` 的对应真实输入/对照声明。
- 操作与验收：`SKILL.md`、`references/induction.md`、`references/hplus.md`、
  本文、`run_review_eight_demo.py`、`review_eight_demo.json`。

## 普通 A/B 到 C 的正常调用

可直接运行，无求解器依赖；默认临时库，也可传一个空隔离目录：

```bash
PYTHONPATH=src python docs/validation/run_review_eight_demo.py
PYTHONPATH=src python docs/validation/run_review_eight_demo.py --home /tmp/or-eight-demo-new
```

脚本里的正常调用是：

```python
# execute_checked 内部：predict_strategy_outcome -> execute(prediction_id=...)
# -> record -> check_task_result(原题 checker) -> close_episode。
pa, a, closed = execute_checked(h, provider, home, 'A', 1000, 300, 2699,
                               decision=1, maximum=True)
pb, b, _ = execute_checked(h, provider, home, 'B', 641, 130, 90)
job = closed['induction_schedule']['pending_ids'][0]
material = h.induction_material(induction_id=job, execution_ids=[b.execution_id])
submitted = h.induce(induction_id=job, relations=[
    {**RULE, 'evidence': [a.execution_id, b.execution_id]}])
entry = submitted['relations'][0]['saved']
pc, c, _ = execute_checked(h, provider, home, 'C', 812, 200, 196, entry=entry)
pd, boundary, _ = execute_checked(h, provider, home, 'boundary', 12, 0, 0,
                                entry=entry, boundary=True)
assert h.sbank.get(entry).is_published
```

实际输出见 [review_eight_demo.json](review_eight_demo.json)：A/B/C/边界目标值分别为
2699/90/196/0；生效版本 1，学习增量标签 1。文件还保存真实生成的预测、归纳动作、
材料、原题/独立执行与回执引用，以及下一次实际发送给替身 provider 的片段。

```json
{"prediction_context":{"hplus_feedback":{"settled_examples":[{
  "decision":1,"label":1,
  "verification_state":"independently_validated_increment",
  "rule_versions":[{"entry_id":"1","version":1}]
}]}}}
```

这个片段只是删去其他字段后的展示；完整原始内容在 JSON 中。
外层归纳 agent 仍负责比较机制、选择材料、提交规则或空回执，以及提供有来源的
适用性/方法/覆盖评审；框架负责授权、引用、版本、重试和发布状态。

## 回归命令与结果

```bash
PYTHONPATH=src:tests/harness python -m unittest -q test_rule_versions test_hplus_binary test_episode_closeout test_evidence_window test_snapshot_lifecycle test_capability_trace test_calibration_v2 test_r20_induction_chain test_review_eight
PYTHONPATH=src:tests/harness python -m unittest discover -q -s tests/harness
```

- 最终定向：253 项，通过（59.434 秒，含新增 30 项）。
- 兼容性补查：45 项，通过（36.635 秒）。
- 成本更正及相关兼容性：69 项，通过（5.465 秒）。
- 最终全量：1877 项，通过（204.271 秒，较基线新增 30 项）。

首轮全量 1876 项暴露两处兼容性问题：错误信息前缀及旧整数实验替身缺机制对照。
两者已修正并通过补查，第二轮 1876 项全通过（191.287 秒）。随后对正常
amend-cost 入口补查时发现发布未重算，已修复并增加第 30 项回归；最终版本另跑全量。
保留首轮失败及后续修复说明。
含本机 HTTP 替身的集成测试在允许回环端口的执行环境运行。

## 兼容范围和后续项

旧任务文本/来源、旧手工三态、旧版本采用和已发出的预测快照继续可读。
不普遍判定自然语言等价：可安全确认的格式等价直接处理，复杂语义需要归纳侧的
有来源对照，否则增量未知。冻结引用保证必要事实可追溯；在线证据 bank 的窗口
继续清理，这不意味着所有规则引用存储也有同一容量上限。

G/C/R 求解选择、效用权重和阈值未改。P1-9 冻结库生命周期、P1-10
`evaluate-capability` 跨执行效果归因、P1-11 实验切分/漏判统计仍留待后续。
本轮验证链路与归因机制，不据替身预测或少量整数题宣称真实 H+ 判别力提升。
