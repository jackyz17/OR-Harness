# 归纳、跨题检验与记忆条目改造验收

审查基线：`main@00e19df6b8c68ea4eae5ac3c877334421aad430c`。
验收日期：2026-11-12。结构化结果见 [induction_memory_v2.json](induction_memory_v2.json)，
可重跑的完整示例见 [versioned_induction.py](../../references/examples/versioned_induction.py)。

## 工作树和实测来源

按要求运行了 `git status --short` 和 `git rev-parse --short HEAD`。本环境的
`.git` 是只读占位目录，二者均返回 `not a git repository`。因此通过 GitHub
Git 数据接口核对远程 main，并逐文件对比修改前快照：175 个本地已有受跟踪文件
与基线 blob 完全相同。原有的 12 个本地独有文档/脚本保持原样，不纳入本次提交。
后续提交基于该远程树创建，分支更新使用预期 HEAD 校验，不覆盖未知远程改动。

未找到原 `mdsc_sc003` 实验库。根目录现有库没有执行和策略条目；仓库中的两个
历史 fixture 只有少量记录。本报告使用隔离临时库、真实 Python 整数枚举脚本及
构造的 29 条同题族长旧规则，复现材料膨胀机制。使用本地 hashing embedding
验证索引调用链，没有调用真实预测模型，不据此声称语义检索或工业任务能力提高。

## 实现的常规流程

1. 执行、原题检查、记录并收尾。已有规则试用在证据窗口淘汰之前冻结检查事实。
2. 实际 H+ 判定为 `allow` 才正常取材；`skip` / `pending` 在检索和展开之前返回。
   未接入实际判定时，显式归纳使用 `explicit_induction_transition` 并记录来源。
   求解前 `capability_gain` 保持预测；本轮没有新 H+ 公式或阈值。
3. 一次 `induction-material --task T` 默认返回受总预算控制的小对照包。建模视角优先
   跨底题机制对照，实现视角优先失败修复；保留实际方法与计划的区别、来源和缺口。
4. 一次 `induce --relation` 提交 `when` / `do` / `expect` 及真实证据 ID，得到候验版本。
   补证据、修订、合并显式选择原条目。完整相同提交幂等，语义相似只提供候选。
5. 冻结规则之后的独立迁移、前提变化边界，以及修订涉及的历史成功回归均通过后，
   切换一个常规生效版本。可信域内反例可暂停生效版本。未知采用/覆盖/归因保留诊断。

```bash
orx close-episode --task T --episode E --terminal completed \
  --induction-decision allow --induction-reason "检验可复用机制"
orx induction-material --task T
orx induce --relation '{"when":"固定 Y 时整数 X 的可行域为区间，目标关于 X 单调。","do":"枚举 Y，计算 X 两端点，按系数符号和最大化或最小化方向选端点，再比较目标；不满足单调性时停用并枚举完整可行集。","expect":"保持原题整数最优值和声明的约束。","evidence":["真实执行ID"],"material_id":"上次取材返回的ID"}'
orx inspect --bank strategic --entry 1 --rule-version candidate
# 定向试用候验规则时，在原 execute 命令中声明 --used-entry-ids 1@1。
```

可直接重跑的验收不需要手填证据 ID：

```bash
PYTHONPATH=src python references/examples/versioned_induction.py \
  --home /tmp/or-induction-new-empty-home --mass \
  --report /tmp/or-induction-report.json
```

`--home` 必须为空目录。若有基线 checkout，可追加
`--baseline-root /path/to/baseline` 同时测量旧材料构造器和旧 CLI 包装。

## 完整默认返回的预算

同一个隔离 fixture；按紧凑 JSON 完整 CLI 输出计数，包含 `result` / `summary`
包装、历史、旧规则、H+、诊断、计数与序列化开销，不含末尾换行。

| 项目 | 改造前 | 改造后 |
|---|---:|---:|
| 完整字符数 | 1,118,044 | 21,769 |
| UTF-8 字节数 | 1,118,056 | 21,769 |
| 返回的 `chars_used` | 3,194 | 21,769 |
| `existing_knowledge` 字符数（改造前） | 1,099,273 | 使用短规则及引用 |

新包低于 32,000 字符预算，保留本题 1 行、相关执行 3 行和短旧规则 8 条，
另有 23 条旧规则通过计数与游标表示；关键生效规则 `1` 没有被最近的长旧条目挤掉。
长方法、题面、规则按完整值引用，不以误导性前缀截断。详情可按执行 ID、条目 ID
或 `value_ref` 读取。旧规则的完整检查/采用日志不再默认内联。

改造前测量调用修改前快照的 `maintenance.py` 构造器和 CLI 总结，底层使用兼容的
当前 bank 读取同一个 fixture。这是构造器/包装层的配对复现，不是完整旧版本进程
的性能测试，也不是原实验库的 814 KB 复测。时间戳长度可使重跑相差少量字符。

## 短规则及真实检验

隔离库条目 `1@1`：

- **when**：X、Y 为非负整数；固定 Y 后 X 可行域为区间，目标关于 X 单调。
- **do**：枚举 Y，求 X 上下端点，根据目标方向和系数符号选择端点；不满足单调性
  时停用降维，转为完整可行集枚举。
- **expect**：保留原题整数最优值及全部声明约束。

| 角色 | 实际求解/检查 | 结果 | 规则状态 |
|---|---|---:|---|
| 提炼材料 | max 2X+5Y，X+Y≤1000，X≥2Y+300 | 2699 | candidate |
| 独立迁移 | min 2X−5Y，X+Y≤641，X≥2Y+130；取下端点 | 90 | 检查 passed，仍 candidate |
| 边界测试 | min (X−3)²+Y，X+Y≤12，X≥2Y；停用降维 | 0 | 检查 passed，转 active |
| 错误规则反证 | 独立最小化例忠实采用“总取上端点”的另一规则 | 92，原题检查 failed | refuted，未发布 |

实际执行 ID、完整检查、采用冻结引用、检查范围、代码及结果工件摘要保存在 JSON。
迁移检查 ID 为
`rc:38193873a8dafd50c60a1119b199817d40f1345bc1250e840e28a9d5702b3c3f`。
收尾读取 `OR_ACTION_ID` 绑定的实际方法回执和 `check-task.rule_assessments`，
不是根据步骤文字相同推定忠实执行。

断言中的前提、忠实度、效果覆盖由有来源的审阅记录及执行工件支撑。示例检查独立
整数最优值、声明整数变量、给定 Y 时 X 的两条约束，审阅记录核对实际枚举代码及
非负 Y/方向。通用 checker 的 `unchecked` 提示仍保存。这不是自动自然语言证明器。

已展示给提炼 agent 的历史案例、相同原题内容、已知同底题 seed/分支均不能充当
独立迁移。成本优势要求同题、同实际求解条件、等价验收和同计量范围的实测对照；
单独一次通过不证明成本优势。技术实现规则只使用明确的执行/协议事实。

## 版本、幂等与兼容性

正文为不可变引用，检查和使用事件分别追加并关联精确版本。修订建立候验版本，
旧生效版本在新版本验证前继续服务；回退到历史正文仍形成新的按时序版本。
合并保留来源及旧编号，目标及来源成功样本的必要回归满足后才完成切换。
重复提交、重复收尾和迟到检查不重复计算同一次实际使用。失败索引同步可按受影响
编号重试；补证据/使用日志不会重新嵌入未变正文。

新执行仅压缩固定说明和重复方法回执，原始运行值、事实来源/冲突、失败细节、
成本掩码/范围、显式零、预测缺失及三态采用声明保持往返。旧 JSON 和历史快照
按原含义读取，常规短视图标明历史状态；没有批量迁移或生产库清理。
旧 `induce --relation` 输入语法通过适配器写入新候验规则，真实旧条目保留原发布语义。
G/C/R 的预测契约、效用公式及在线门控未改动。

## 回归验证

最终源码的定向回归：**193 项通过，0 失败**。

```bash
PYTHONPATH=src:tests/harness python -m unittest \
  test_rule_versions test_episode_closeout test_evidence_window \
  test_calibration_v2 test_capability_trace test_snapshot_lifecycle
PYTHONPATH=src:tests/harness python -m unittest discover -s tests/harness
```

最终 `tests/harness` 全量：**1831 项通过，0 失败、0 错误，耗时 161.570 秒**。
覆盖普通 CLI 收尾/取材/短提交，H+ 提前跳过/待定，总返回预算，独立迁移/边界，
同底题去重，候验/生效/暂停，修订回归，合并，幂等，精确版本采用，迟到检查，
索引失败重试，立即淘汰前冻结检查，以及历史 JSON/快照与 G/C/R/H+ 原有回归。
旧行为测试按真正历史 payload 构造 fixture；新提交期望更新为候验发布，未跳过测试。

## 仍需实际证据的部分

原实验库缺失，因此没有复测其 26/29 单题条目或约 99% 缺失实际方法的比例。
旧执行缺少实际方法时继续显示未知，不根据计划补造事实。未核实的底题关系、
规则必要前提、忠实采用和 checker 效果覆盖不能自动变成迁移证据。
后续需用自然发生的独立工业任务检验迁移范围、边界和收益；本轮结果证明机制接通。

## 修改文件

实现：`src/or_harness/api.py`、`cli.py`；`core/{schema,storage,evidence_encoding}.py`；
`strategy/{induction,rule_versions,evidence_views,strategic_bank,experience_bank,selector,vector_recall,embedding_index}.py`；
`world_model/{maintenance,induction_material,induction_gate,state,episode_closeout}.py`。

测试：`tests/harness/legacy_rules.py`、`test_rule_versions.py`、`test_admission.py`、
`test_end_to_end_cli.py`、`test_evidence_window.py`、`test_offline_induction.py`、
`test_r10_end_to_end.py`、`test_r10_fixes.py`、`test_r14_induction.py`、
`test_r16_assertion_symmetry.py`、`test_r20_induction_chain.py`、`test_r21_utility_management.py`、
`test_relations.py`、`test_verification_honesty.py`、`test_world_model.py`、
`test_world_model_m6.py`、`test_world_model_m6_closeout.py`。

说明及示例：`SKILL.md`、`README.md`；
`references/{commands,concepts,episode_closeout,induction,prediction_context,world_model_contract}.md`；
`references/examples/{no_catalog,versioned_induction}.py`；本报告与对应 JSON。
