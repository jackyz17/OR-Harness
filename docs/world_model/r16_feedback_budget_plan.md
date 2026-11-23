# r16：世界模型历史反馈压缩与预测输入预算（第一阶段）

基准 `aef3a9e`（main）。本轮**只改最终投递给世界模型的视图与请求预算**：
完整历史事实、校准数据库、原始预测/观测/误差/归因全部保留，不新增任何
LLM 摘要调用，不修改 G/C/R 定义、H+、知识适用性、归纳接口、U 计算和
预测串并行。

## 0. 事实核对

build3 的 12 条 paired feedback 占约 94KB。用本地 12 条真实闭环样本复现
（`tests/harness` 决策夹具，同一固定输入）：

| 指标 | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| `prediction_execution_pairs` 块字符 | 49,420 | 27,977 | −43.4% |
| provider context 字符 | 84,905 | 64,897 | −23.6% |
| JSON body 字符 | 89,924 | 71,186 | −20.8% |
| 整请求字符（system prompt + body） | 112,498 | 92,490 | −17.8% |
| 整请求 token（保守估计） | 28,124（旧 `chars/4` 估计） | 30,853 | 标注口径不同，均 < 55,808 |
| 12 条中保留案例数 | 12 | 12（默认历史预算 30,000 字符） | 0 丢失 |

固定合成输入的压缩对比（12 条“重”历史，字段完整、长预测注记、验收规格）：

| 输入 | 压缩前 | 压缩后 | 保留案例 | 逐字段缩减 |
|---|---:|---:|---:|---:|
| 0 条 | 2,324 | 1,094 | 0 | 0 |
| 1 条 | 5,945 | 5,163 | 1 | 0 |
| 12 条（默认 30,000 上限） | 49,413 | 28,532 | 9 / 12 | 20 |
| 12 条（上限 40,000） | 49,413 | 35,914 | 12 / 12 | 20 |

12 条默认上限时被省略的 3 条**全部带 `dropped_pair_refs`**（`orx inspect
--bank evaluations --evaluation <id>`），其余 9 条逐行删减的字段列在
`provider_compaction.fields_reduced`（自由文本/审计字段，预测、观测、误差、
口径、eligibility 不删）。

## 1. Provider-only 历史反馈视图

`build_paired_feedback` 发布期产出的完整 block 与选择逻辑不变；新增
`evidence_views.provider_paired_feedback`/`provider_feedback_pair`，只在
`PredictionContext.provider_view()` 读取时派生，不回写。

每条保留：相关条件（task/cell/family/strategy）、计划/实际方法（步骤全量
或引用 + 已记录差异）、原始预测（G predicted/interval、C predicted、R 概率）、
真实观测（G observed、C actual、task_check、R label）、误差（signed_error、
abs_error、log ratio、Brier）、已记录的归因（`method_fidelity`/
`failure_source`/`source`/assessment 缩写 + ref）、证据引用与 `detail_ref`。

同时保留：G/C/R 的 metric 与 observation rule 版本、`eligibility`、
`label_eligibility`、unknown、blocked/comparability 事实；crash 行没有
`observed`（绝不写 0），unknown label 保持 `null`，不把
`execution_status=error` 变成 G=0。

确定性提取：所有值来自现存结构化字段；仅做字符串截断、列表限量、引用改写，
不做语义判断，不根据 check 失败推断策略错误原因。

## 2. 三级独立体积预算

| 视图 | env | 默认 | 缩减顺序 |
|---|---|---:|---|
| 历史反馈 | `OR_HARNESS_HISTORY_FEEDBACK_CHARS` | 30,000 | 先逐行删可选明细 → 再从尾部逐条丢弃（选择顺序已按 labeled/relevance 排序）→ 最后仅保留计数摘要 |
| G/C/R 证据（`cost_evidence` + `implementation_r_evidence`） | `OR_HARNESS_GCR_EVIDENCE_CHARS` | 16,000 | 删重复 `event_definitions`、逐组/逐 case/逐 range 尾部 |
| 其他非核心（capability/version/retrieval/H+/reminders/notes） | `OR_HARNESS_NONCORE_CHARS` | 20,000 | 最大块先按结构缩减 |

`trim_history_block` 全程报告 `n_pairs_included`/`n_pairs_omitted`/
`n_pairs_provider_dropped`/`dropped_pair_refs`/`omission_note`；摘要形态仍带
`detail_ref: orx calibration`。绝不静默留空。

## 3. 整请求预算

新增 `world_model/request_budget.py`：

- 量的是 **system prompt + 完整 JSON body**，并计入 candidate、context、
  contract 等所有实际发送字段；
- provider 暴露 `count_tokens` 时用**真实 token 数**（`provider.py` 新增可选
  contract 方法，基类返回 `None`）；否则用明确标记的保守估计：
  `tokens = ascii/3 + cjk*1.2 + other`，向上取整；CJK/JSON 密度高于旧的
  `chars/4`，不再低估；
- 超限时按 **历史反馈 → 其他非核心 → G/C/R 证据** 顺序缩减（`trimming`
  逐阶段记录 before/after）；当前任务正文、候选步骤、`execution_constraints`、
  `strategy_outcome_calibration` 与预测契约（risk/benefit/cost/output/
  hplus/protocol）**不截断**；
- 若受保护核心本身超预算：`sendable=false`，`StrategyOutcomeService.predict`
  **不调用 provider**，持久化 `status=invalid` 的预测并记录
  `failure.kind=request_over_budget` 与尺寸诊断；绝不发送无控制请求。

`context._bound_request` 的裁剪顺序同步修正：历史反馈现在最先被逐行缩减，
不再先删当前任务相关证据；`solving_context` 从可裁剪列表移除并受保护。

## 4. 修改位置

- `src/or_harness/strategy/evidence_views.py`：provider-only 压缩、三级预算、
  省略引用。
- `src/or_harness/world_model/request_budget.py`（新）：token 测量、整请求
  预算、分阶段缩减与拒绝发送判定。
- `src/or_harness/world_model/context.py`：`provider_view()` 接入压缩视图与
  独立预算；`_bound_request` 裁剪顺序与保护块。
- `src/or_harness/world_model/strategy_prediction.py`：`_finalize_request_size`
  改为整请求预算；`predict()` 超预算拒发并记录失败。
- `src/or_harness/world_model/provider.py`：可选 `count_tokens` 接口。
- `tests/harness/test_history_feedback_budget.py`（新）：压缩对应性、
  0/1/12/大量、CJK/JSON token 密度、整请求优先级、固定输入 before/after、
  端到端“历史先缩/核心不删/超限拒发”。
- `references/world_model_contract.md`、`references/episode_closeout.md`：
  请求视图与预算说明。

## 5. 验证

```bash
TMPDIR=$PWD/.tmp_tests PYTHONPATH=src:tests/harness python -m unittest \
  tests.harness.test_history_feedback_budget \
  tests.harness.test_prediction_context tests.harness.test_memory_storage \
  tests.harness.test_g_decoupling tests.harness.test_wm_calibration_loop \
  tests.harness.test_episode_closeout tests.harness.test_r15_recall_pairing \
  tests.harness.test_strategy_outcome tests.harness.test_calibration_v2 \
  tests.harness.test_r_technical
```

结果见提交说明。全部为定向测试，未跑全量、未重建数据库、无真实模型调用。

## 6. 剩余风险

- 无 provider tokenizer 时 token 数是保守估计；对 CJK 可能偏大（更安全但会
  更早触发缩减/拒发）。
- 历史预算（默认 30,000 字符）是展示上限，不是校准样本上限；默认下过重的
  12 条可能只显示 9 条，其余以引用报告，需要时调大 env。
- 被逐行缩减的自由文本（why/fallback/assessment/notes/条件细节）只保留引用
  恢复，模型看到的是压缩文本。
- legacy `world_model/service.py` 的知识预测请求仍只做 context 片段预算，
  尚未接入整请求拒发（本轮范围是世界模型策略结果预测请求）。
