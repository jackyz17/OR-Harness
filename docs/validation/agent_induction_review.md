# Agent-led induction、效用管理和 H+ 验收

评审基线：`main@40581f75277365d49adcd787162d064165c4031f`。
本轮按更新后的要求替换该提交中强制发布、材料缓存及自动状态迁移的行为。
`review_eight.md` / `review_eight_demo.json` 保留为该旧提交的历史验收，现行用法以
本文、`SKILL.md`、`references/induction.md`、`references/hplus.md` 和命令说明为准。

服务器工作目录没有可用 Git 元数据：实际运行 `git status --short`、
`git rev-parse --short HEAD` 均返回 `not a git repository`。通过 GitHub 连接读取
实际 main、提交和树，核对开始前的本地文件哈希；没有把远程 HEAD 冒称本地 HEAD。
保留本地已有文档和未涉及的远程文件。测试和整数枚举均在本服务器执行。

## 行为变更与删除的机制

| 范围 | 当前行为 | 删除或替换 |
|---|---|---|
| 发布和迁移 | agent 用 `publication:{state,reason,scope}` 选择发布或候验；正常检索展示正文、实际检查及其范围 | transfer/boundary/regression 统一发布门槛；基于题面数字、题型、路径或母题字段的迁移裁决 |
| 修订和效用 | 同 ID 保存不可变版本；候验期间保留生效版；agent 显式切换、合并，按真实采用版本进行 promote/demote/keep/retire | 旧成功执行的强制回归待办；版本化管理请求只记诊断；已有条目管理依赖正待办或空归纳回执 |
| 取材和证据窗口 | 每次即时查询；实际触发执行固定，较新的尝试正常分页；仅待办所需完整证据临时保留，结清后释放 | 查询键整包缓存及回放；普通 check-task 另存完整执行；旧共享证据把旧提交派给新待办 |
| 成本和更正 | agent 选择两次执行和维度；框架提供方法、条件、验收、范围、测量来源和逐维变化；排除/恢复更新有效性 | 自动搜索/选定基线、替换承诺维度、裁定成本优势；事实更正触发全量重验及自动暂停、恢复、发布 |
| H+ | 冻结的二值开关不变；`wm-hplus-review/2` 使用 agent 的学习结论及有效支持，区分假设和增量 | 正文哈希差异、旧库漏召回/为空、当前发布或退役状态推导标签；长反馈塞入整段比较，或首条放不下即停止 |

新正文未选择发布时保持候验；这只是默认选择，不是内容审判。旧字段的任务数仍作
事实报告，`required_checks:[]`、`authority:agent` 明确说明没有统一检查门槛。
原题引用和 checker 版本、已发出的预测输入、真实采用的正文版本保持可追溯。
实际失败保留为事实，由 agent 判断效用和归因；不是自动撤销知识的信号。

## 五组验收实跑

针对更新后的决策重写 `tests/harness/test_review_eight.py`，共 29 项。
使用隔离库、替身预测器、实际 Python 整数枚举以及正常 CLI/API，不据此宣称真实判别力。

### 1. 冷启动发布与后续 C

`test_A_B_derivation_and_constructed_boundary_publish_before_real_C`：
A 的 X+Y≤1000、X≥2Y+300、最大化 2X+5Y 得 **2699**；
B 的 X+Y≤641、X≥2Y+130、最小化 2X−5Y 得 **90**。
agent 给出端点推导，以真实穷举和非单调构造边界（最优值 **0**）核对，
在没有真实 C 时选择发布，规则立即可检索，H+ 为 **1**。
后来真实 C（812/200）实际采用版本 1 得 **196**，登记为 agent 所选的迁移观察。

`test_renamed_same_input_and_one_sided_parent_are_advisory_not_a_gate`：
改编号、保留原始数据和模型的改写、单侧 parent 来源均显示已知重合；用途仍由
agent 决定，没有框架自动拒绝发布/迁移。
`test_one_unreviewed_observation_stays_candidate_even_after_more_checks`：
单次未验观察保留候验，后来增加 C/边界检查也不会自动发布或自动给出学习正例。

### 2. 低成本核验、修订、补证据与合并

`test_reference_only_review_preserves_scope_and_versions_need_explicit_activation`：
A 只有 `reference_objective:2699` 时，正常 check-task 可记录数值一致；范围仅到
该项。v1 历史事实可评估 v2 兼容，标为 historical_review，不能冒充 actual_version_use。
v2 实际执行才有 v2 use；候验期间 v1 仍可检索，显式发布才切到 v2。
弱化后的 checker 有不同指纹，原检查由 previous_check_ref 保留；没有自动回归待办。

`test_evidence_only_and_publication_are_light_maintenance_with_no_new_job`、
`test_agent_confirms_merge_without_compulsory_source_regressions`：纯补证据复用正文版本；
已有候验版本的发布选择可走维护入口；agent 确认目标和来源后可合并，没有逐项旧执行门槛。

### 3. H+=0 效用管理及成本事实

`test_H0_management_promote_demote_keep_and_retire_share_version_lifecycle`：
真实采用的 H+=0 执行可经 `induce --attribute-effect` / API 立即 demote、keep、
promote，状态与正常检索一致；retire 进入既有冷归档。没有新增归纳动作或空归纳回执。
`test_management_of_old_use_does_not_demote_a_later_active_version`：v1 使用的后续
管理只作用于 v1，不能静默暂停已生效的 v2。

`test_no_automatic_cost_baseline_and_explicit_contrast_reports_both_dimensions`：
两次实际方法的测试宿主用量为调用 **9→10**、tokens **100→25**。
不指定基线时没有自动比较；显式选定后分别返回 **+1 调用、−75 tokens**、验收、
条件和来源，`verdict:null`，没有“调用减少”或整体省成本的框架结论。
`test_estimates_are_visible_but_never_converted_to_measured_cost_evidence`：
`agent_estimate` 的 100→1 tokens 仍标为 estimate，不成为实测收益。

### 4. 即时查询、重试、并发和窗口

`test_positive_anchor_live_queries_and_pagination_include_newer_zero_attempt`：
同题 A=1 后又执行 B=0/C=0，主例始终是 A；同查询重读得到当前材料，控制案例分页
能遍历较新的 B/C，没有整包 hplus_material 写入。

`test_pending_selected_evidence_survives_window_restart_then_releases_full_holds`：
小窗口淘汰 A 和已选对照，重启后仍能正常取材/引用，提交后完整临时保留数为 **0**；
最小审计事实可读取原题输入引用和 checker，原 bank 行未回填。
`test_shared_selected_evidence_stays_held_until_both_pending_reviews_finish`：
共享对照在另一待办未结清前不释放。
`test_ordinary_recheck_omission_preserves_assessment_and_no_full_row_copy`：
未被引用的普通执行检查不产生完整 rule_execution / induction_execution 副本。

`test_old_shared_evidence_submission_retries_original_receipt_not_new_job`：
跨轮共用 B 的旧提交重试返回原回执，新待办仍待处理。
`test_two_connections_late_material_never_resets_completed_review`：两连接交错最终
只有一项真实归纳动作和一份结果，迟到材料不重开待办。
重复收尾复用实际绑定；多次真实正信号分别结清，有歧义才选择已有对象。

### 5. H+、长反馈及更正

`test_supported_increment_is_independent_of_publication_and_retirement`：有来源的
机制增量可以在候验时得到 1；后来因效用退役不能单独推翻原学习结果。
`test_empty_recall_and_synonyms_never_supply_a_positive_by_themselves`：
漏召回/空库、同义改写、发布均不会自行给 1。
`test_safe_formatting_dedup_and_evidence_only_supported_negative`：安全格式去重；
有依据的已有覆盖加补证据可为 0；缺支持为未知。

`test_exclusion_invalidates_only_future_learning_support_not_publication_or_old_input`：
排除唯一有效支持后，后续 H+ 变未知，条目仍由 agent 管理；原输入与旧 cutoff 可复读，
恢复后重新读取有效来源。正常检索中的证据范围也标出当前有效性。
`test_late_checker_enters_only_later_negative_observations`：迟到检查只影响之后反馈。
`test_long_review_is_compact_and_short_conclusion_reaches_actual_next_provider` 与
`test_oversized_first_feedback_does_not_hide_a_later_fitting_receipt`：
长比较只给短结论和引用，仍放不下时继续尝试后续回执；下一次实际 provider 请求能收到短例。

## 最短正常调用示例

完整可运行实现（默认临时库，可传空目录 `--home`）：

```bash
PYTHONPATH=src python docs/validation/run_review_eight_demo.py
```

核心调用如下；`execute_checked` 实际执行 predict → execute → record → check-task → close：

```python
pa, a, closed = execute_checked(h, provider, home, 'A', 1000, 300, 2699,
                               decision=1, maximum=True)
pb, b, _ = execute_checked(h, provider, home, 'B', 641, 130, 90)
_, boundary, _ = execute_checked(h, provider, home, 'constructed-boundary',
                                12, 0, 0, boundary=True)
job = closed['induction_schedule']['pending_ids'][0]
h.induction_material(induction_id=job,
                     execution_ids=[b.execution_id, boundary.execution_id])
submitted = h.induce(induction_id=job, relations=[{
    **RULE, 'evidence': [a.execution_id, b.execution_id, boundary.execution_id],
    'publication': {'state': 'published', 'reason': '端点推导、枚举与停用边界已核对',
                    'scope': ['所述单调整数区间机制及列出的检查']}}],
    review={'outcome': 'new_rule', 'comparison': {
        'sufficient': True, 'substantive': True, 'basis': 'derivation',
        'source': 'agent 核对原题、推导与实际枚举',
        'assessment': '固定 Y 后由优化方向和单调性选端点；非单调时停用。',
        'evidence_refs': [a.execution_id, b.execution_id, boundary.execution_id]}})
entry = submitted['relations'][0]['saved']
assert h.sbank.get(entry).is_published  # 此时还没有真实 C
_, c, _ = execute_checked(h, provider, home, 'C', 812, 200, 196, entry=entry)
h.induce(effect_attributions=[{
    'entry_id': entry, 'execution_id': c.execution_id,
    'verdict': 'helped', 'lifecycle_action': 'keep',
    'note': 'agent 核对实际 v1 使用和原题检查；没有主张成本优势。'}])
```

[agent_induction_demo.json](agent_induction_demo.json) 保存实际生成的 ID、版本、回执、
发布范围和下一次真正发给替身 provider 的请求片段。实跑输出包括：

```json
{"objectives":{"A":2699,"B":90,"C":196,"boundary":0},
 "published_before_real_C":true,"active_version":1,"hplus_label":1,
 "management":{"maintenance_only":true,"version":1,"action":"keep","new_induction_actions":0}}
```

## 文件与测试

- 正常入口：`src/or_harness/api.py`、`cli.py`、`core/schema.py`。
- 规则/效用/证据：`strategy/rule_versions.py`、`strategic_bank.py`、`induction.py`、
  `evidence_views.py`、`evidence_access.py`、`rule_support.py`。
- 取材/反馈：`world_model/induction_material.py`、`hplus.py`、`hplus_support.py`、
  `strategy_prediction.py` 的联合运行时提示词。
- 主回归：`test_review_eight.py`、`review_fixtures.py`、`test_rule_versions.py`、
  `test_hplus_binary.py`。相关历史发布/帮助断言在 `test_offline_induction.py`、
  `test_r10_end_to_end.py`、`test_r10_fixes.py`、`test_relations.py`、`test_r7_semantics.py`、
  `test_r16_assertion_symmetry.py`、`test_r17_hplus_followup.py` 同步更新。
- 文档与示例：`SKILL.md`、`references/induction.md`、`hplus.md`、`commands.md`、
  本文、更新后的 `run_review_eight_demo.py` 和新示例 JSON。

```bash
PYTHONPATH=src:tests/harness python -m unittest -q test_review_eight test_rule_versions test_hplus_binary test_episode_closeout test_evidence_window test_induction_material test_r20_induction_chain
PYTHONPATH=src:tests/harness python -m unittest discover -q -s tests/harness
```

最终交付验证：定向 **155 项通过（40.972 秒）**；
`tests/harness` 全量 **1876 项通过（192.095 秒）**。本轮更新后的主场景 29 项。
已完成的其他验证：
定向 155 项通过（40.702 秒），兼容性 128 项通过（17.625 秒），
新语义及效用专项 81 项通过（25.523 秒），帮助/文档 20 项通过（0.248 秒）。
旧能力归因、快照与新归纳联合补查 153 项通过（22.257 秒）。
首轮全量暴露 9 个旧接口/断言问题，第二轮只剩旧文档门槛断言，均已逐项修正；
清理旧 helper 时还发现 legacy 效果归因保留的导入依赖，已恢复其兼容实现；
新版发布不调用它。不把这些中间失败记作最终通过。

外层 agent 仍负责归纳推理、证据是否充分、发布范围、效用归因与管理选择；框架核对
真实引用、执行/检查、版本与有效性，不自动证明自然语言或数学结论。
G/C/R 求解决策、效用权重及阈值未改。原 P1-9 冻结库生命周期、P1-10
`evaluate-capability` 跨执行效果归因、P1-11 实验切分和漏判统计仍在后续批次。
历史实验字段名不能当作本轮新标签的独立验证结论；本轮替身测试只报告链路验证。
