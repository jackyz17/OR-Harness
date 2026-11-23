# r17：Strategic Knowledge 语义适用性（第二阶段）

基准 `5e98225`（main，含历史反馈压缩与 P1 修复）。本轮只重新明确 Strategic
Knowledge 的适用性语义与读取路径；不改 G/C/R 定义、H+/U、归纳接口、知识证据
内容或已有规则版本/发布/验证/执行引用。

## 0. 设计决定

1. `when` 是规则完整适用条件的唯一语义来源，由归纳 Agent 负责；框架不判断
   抽象语义条件。
2. 框架只做确定性结构检查（family 标签与耦合 cell）、检索、证据管理。
3. 不新增谓词分类体系；`predicates` 变为可选机器条件，新规则不要求提交。
4. 空 `predicates` 表示“没有可确认的机器结构条件”，**不是无条件适用**。
5. 检索命中、机器结构兼容、Agent 采纳严格分层。

## 1. 三层语义模型

| 层 | 产出 | 含义 |
|---|---|---|
| 检索命中 | `similarity`、`channels`、`evidence_class` | 文本/结构相似，值得阅读；不表示适用、不表示已采纳 |
| 机器兼容 | `machine_applicability.state` = `compatible` / `partial` / `unknown` / `not_specified` / `conflict`，含 `checked_conditions`、`unknown_fields`、`conflicts`、`family_label` | 仅确定性结构检查；`compatible` 是**必要非充分**条件；`not_specified` 绝不等于适用；`conflict` 只由“已测量的结构值落在规则 cell 之外”产生 |
| Agent 采纳 | 执行记录 `used_entries` / 版本绑定 / effect attribution | 求解 Agent 读到 `when` 后自行决定；召回命中与机器兼容都不写采纳 |

补充：

- `family` 标签不同只记录为 `family_label.relation=different`（弱信号），
  **不判 conflict**；共享数学机制可以跨场景迁移，由 Agent 依据 `when` 判断。
- 语义命中携带 `semantic_applicability = {state: requires_agent_assessment,
  when: ...}`、`requires_agent_assessment=true`、`adoption_state=not_adopted`。
- 规则正文 `rule_body`（when/do/expect）随知识 hit 与结构推荐一起投递；
  世界模型与求解 Agent 看到的是规则本身，而不只是相似度/统计/谓词。
- 历史 `predicates` 继续读取；规则修订省略该字段时沿用 prior predicates，
  新规则为空集。

## 2. 修改位置

- `src/or_harness/strategy/applicability.py`（新）：`machine_applicability`、
  `semantic_applicability`、`applicability_annotations`、`rule_body`、
  `machine_matched`/`possibly_applicable`。
- `src/or_harness/strategy/vector_recall.py`：`classify_applicability` 改为机器
  分层；知识 hit 增加 `rule_body`/`machine_applicability`/
  `semantic_applicability`/`requires_agent_assessment`/`adoption_state`；
  `structural_match`/`reusable` 保留字段名但改为严格机器语义（空集不是匹配）。
- `src/or_harness/strategy/selector.py`：`Recommendation` 增加上述字段；
  `_from_entry` 附机器/语义分层与警告；结构召回不再把空谓词/未知值当适用。
- `src/or_harness/strategy/strategic_bank.py`：`matching` 默认返回“未被确认
  结构冲突”的条目（保持可发现性），`compatible_only=True` 供取用条目估计的
  严格路径。
- `src/or_harness/core/schema.py`：`StrategicEntry.matches` 改为严格机器匹配
  （空集 False）；`profile_matches` 文档明确其只是机器检查。
- `src/or_harness/api.py`：成本估计的 entry 路径要求机器兼容。
- `src/or_harness/world_model/context.py`：结构 hit 与 provider 视图携带规则
  正文/分层适用性；证据分类不以机器兼容冒充 verified。
- `src/or_harness/world_model/maintenance.py`：归纳材料透传分层适用性字段。
- `src/or_harness/strategy/rule_versions.py`：修订省略 predicates 时保留 prior。
- 文档：`references/world_model_contract.md`、`references/induction.md`、
  `references/concepts.md`、`src/or_harness/world_model/state.py`。
- 测试：`tests/harness/test_applicability_semantics.py`（新）及受语义变更影响
  的既有断言更新。

## 3. 回归测试覆盖

- 空谓词 / 非空谓词 / 未知结构值 / 已知结构冲突；
- 跨 family 且共同数学机制可被发现（不被 label 否决）；
- 语义命中不自动记为已采纳/已验证适用；
- 规则采纳与实际执行的版本绑定（既有 `used_entries`/版本测试保持通过）；
- 历史规则读取与 published/verification 生命周期兼容；
- 整数数量规则 vs 连续数量规则：两者都可被发现、`when` 相反、机器层都
  `not_specified`，由 Agent 判断采纳。

## 4. 剩余风险

- `profile_matches` 仍用于证据窗口/可用性过滤（空集在该处为空约束）；它是
  机器过滤，不是适用性。适用性代码统一走 `applicability` 模块。
- `structural_match`/`reusable` 字段名保留以兼容旧读取方，但语义已收窄为
  机器兼容；旧归档上下文中的值保持不变（历史即事实），新请求使用新语义。
- 跨 family 仍可被“已测量结构冲突”排除；这是确定性矛盾，不是标签判断。

## 5. 验证（定向）

```bash
TMPDIR=$PWD/.tmp_tests PYTHONPATH=src:tests/harness python -m unittest \
  tests.harness.test_applicability_semantics \
  tests.harness.test_vector_recall tests.harness.test_banks \
  tests.harness.test_prediction_context \
  tests.harness.test_induction_material tests.harness.test_rule_versions \
  tests.harness.test_memory_storage tests.harness.test_numbered_entries \
  tests.harness.test_r15_recall_pairing tests.harness.test_cost_round \
  tests.harness.test_admission tests.harness.test_verification_honesty \
  tests.harness.test_mechanism_layer tests.harness.test_wm_calibration_loop \
  tests.harness.test_g_decoupling tests.harness.test_episode_closeout
```

实测：新测试 13 项；上述受影响范围合计 201 + 152 + 98 + 78 + 100 + 64
项通过。唯一失败与基线一致：`test_documented_result_keys.test_inspect_capability`
的既有 CLI `--protocol` argparse 错误（`aef3a9e`/`5e98225` 未改动路径）。
