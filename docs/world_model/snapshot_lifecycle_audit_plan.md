# 完整快照生命周期：本地审计与修改计划

后续实现及验收见 [实现记录](snapshot_lifecycle_implementation.md)。本文保留审计时的基线和发现。

审计日期：2026-10-15。本文记录最初“审计代码并给出修改计划”阶段的发现和拟议工作；下文的未实施、未修改等状态均指该阶段。用户随后已授权实施，完成情况见顶部链接的实现记录。

## 1. 实际基线与边界

- 工作目录：`/data/home/zhanyiqi/or-experience-bank`。
- `.git/` 当前为空，`git status --short` 返回 `not a git repository`。实际分支、HEAD、既有未提交差异均无法核实。未初始化仓库、未重置文件、未连接远端。
- 用户指定基线为 `world-model-integration` / `f67e9a85f88e059344cde5bbb0f60a5ca0acecd9` / r23；本地还有 `test_r24_archive_scan.py` 和相应 task-scoped execution→action 索引实现，不能宣称本地恰好是 r23。
- 本地 49 个 `src/or_harness/**/*.py` 的内容指纹：`760b2654e43c0dd44326fe5ac4e34c93ba5117ec9487687b91c83975f04a0ed9`。算法：按相对路径排序，依次向 SHA-256 输入路径 UTF-8、NUL、文件原始字节、NUL。后续执行可用此指纹核对审计版本。
- 未发现适用的 `AGENTS.md`。`harness_review_bench/` 为空，未获得提示词所述真实大库导出。没有读取或清理真实主库、DS 库；复现和测试使用临时目录。
- 用户报告的主库约 1.30 GB、快照约 842 MB / 3,609 行、上下文约 374 MB / 719 行，均为用户报告，非本轮实测。
- 保持两层记忆；不改 strategy_id、分组/分箱、Q/C/R 定义、策略选择公式、归纳门槛、检查标准和基模。`why` 参与策略身份的问题仅登记为独立问题。

## 2. 已确认的问题

| 优先级 | 代码证据 | 结论及影响 |
| --- | --- | --- |
| P0 | `api.py:snapshot`、`_coverage_view`（385、463 行） | 每次冻结都向 `belief_snapshots` INSERT，包含同一结构格所有策略统计、支持 execution_ids 和匹配知识正文；动作前后都会触发，生命周期未受决策约束。 |
| P0 | `context.py:build_context`、`snapshot_conditions`（1978、2114 行） | 新上下文再复制 coverage/harness/X/B，`solving_context` 又保存 X/B。重复不仅发生在两张表之间，也发生在上下文内部。 |
| P0 | `api.py:_episode_progress`（345 行） | 为恢复最新进度，读取并反序列化该任务全部快照。直接停写会丢失跨动作进度，直接删除旧表会破坏恢复。 |
| P0 | `api.py:bind_strategy_outcome`（2158 行）、`service.py`（384、423 行） | 问题身份和旧预测的假想防线依赖完整快照；存在直接 SQL，不能只修改 `get_snapshot()`。 |
| P0 | `episode_closeout.py:_frozen_declared_budget`（1281 行） | 预算可比较性读取 `trace.input_snapshot_id`。快照缺失时不再进行该预算比较，可能让预算变化失去阻断。 |
| P0 | `maintenance_decision.py:_episode_execution_span`（1667 行） | H+ 后验从执行前快照知识列表判断当时可用性。必须保留此证据语义，不能用实际采用列表替代。 |
| P1 | `state.py:put_knowledge_version`（86 行）、`api.py:_coverage_view` | 共享知识版本已经存在，但正文仍 inline；当前优化只是增加共享副本，没有消除正文副本。版本对象还是整批知识，不能沿用为新的全库版本仓。 |
| P1 | `episode_closeout.py:calibration_summary_for_context`（4761 行）、`context.py:provider_view`、`_bound_request` | published summary 的 `validity_corrections` 原样进入上下文和模型。模型/scope 过滤只过滤 groups，未去掉修正明细。校准整体不在可裁剪列表中，会先挤掉召回与反馈。 |
| P1 | `api.py:build_prediction_context`（851、1113 行） | `persist=False` 只跳过上下文写入，未传入 snapshot 时仍写完整快照；CLI `context --no-persist` 因而不是完整的无持久化路径。 |
| P1 | `context.py:PredictionContext.from_dict`（1833 行） | 当前严格只接受 `wm-context/3`；列表读取还会跳过不支持版本。直接改版本常量会导致旧上下文拒读或消失。 |
| P1 | `api.py:_record_used_entries`（7356 行） | 能区分未声明与显式 `[]`，但保存的是条目编号和当时是否存在；没有可恢复正文的不可变版本引用。 |
| P1 | `strategy_prediction.py:StrategyOutcomeService.predict/_save`、`api.py:predict_strategy_outcome` | 服务先保存预测，API 随后补 context/问题身份并再次保存。进程在两次保存之间中断会留下不完整关联；现有 `Store.transaction()` 也不能直接当作嵌套 savepoint 使用。 |
| P1 | `episode_closeout.py:archive_calibration_detail`（6310 行） | 已归档评估、预测、上下文，未回收完整快照和旧知识批次。r24 的任务级动作索引已修复部分归档扫描成本，后续必须保留。 |
| P1 | `api.py:record`（6032、6243 行）、`selector.py:recall`（239 行）、`episode_closeout.py:republish_calibration` | record 追加可见证据并同步索引；recall 更新 consulted；closeout 重发布校准及 pairs。仅不执行 induct 无法实现跨题固定记忆测试。未找到完整的冻结测试入口。 |

口径核对：`stats.py:for_profile`（252 行）先以 `group_key` 限定同一结构格，再按 `strategy_id` 聚合。所以统计项数首先应解释为该格策略数，不能说成全库结构格数。单格支持执行编号仍随该策略证据数量增长。

请求体已有两项可复用实现：provider 投影去重且移除内部 CIR；完整请求另有 system prompt + JSON body 的字符估算。不要把它们重新实现一遍，也不要把字符估算说成 tokenizer 实测。

## 3. 临时库复现

固定同一题面与结构格，固定候选 `strategy-000`，每个策略写一条真实执行格式的构造记录；使用未配置模型的 provider，不发模型请求、不运行求解器。构造 1/10/100 个策略，分别保存一次快照和一个当前上下文。召回上限保持默认，非“固定所有召回命中”的等价性测试。

| 同格策略数 | cell_statistics 字节 | 快照 payload 字节 | 上下文 payload 字节 | provider 片段字节 | 完整请求字符估算，含 system prompt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 655 | 3,362 | 18,843 | 17,411 | 38,796 |
| 10 | 6,541 | 9,250 | 27,436 | 20,551 | 42,114 |
| 100 | 65,401 | 68,111 | 86,299 | 20,553 | 42,116 |

- 字节按 `Store.dumps()` 同口径的紧凑 JSON / ASCII 转义 / UTF-8 计算；表中快照与上下文字节还经 SQLite `length(CAST(payload AS BLOB))` 核对。编号和时间变化可能带来少量波动。
- 三组 `ctx.snapshot.coverage == snap.coverage` 均为真；`wm-context/3` roundtrip 可读。
- 每组再执行一次 `build_prediction_context(..., persist=False)`，快照行数由 1 增至 2，上下文保持 1 行；100 策略组额外写入约 68,110 字节快照 payload。
- 向临时上下文注入一条构造 `validity_corrections.detail`，provider 投影仍携带该明细，确认发送路径。
- provider 片段没有随 10→100 策略等比例增长，说明不能用磁盘增长直接推断请求 token 同比例增长。
- 本轮没有实施改造，所以这些是现状基线，不是改造前后对比；没有 VACUUM 或物理空间回收，不承诺真实库压缩比例。
- 本轮诊断脚本及原始结果暂存于 `/tmp/or_harness_snapshot_audit.py` 和 `/tmp/or-harness-snapshot-audit.json`；正式实施时应将稳定夹具纳入回归测试，避免依赖临时文件。

## 4. 消费者 → 最小信息 → 新来源

下列新字段名只是落地建议，优先复用已有 action/trace/execution_features 和关联；不复制同一字段到所有对象。

| 消费者 | 真正需要的最小信息 | 新来源；旧记录回退 |
| --- | --- | --- |
| `_episode_progress`、动作前后状态 | 同题目版本、同 episode 的最新真实 X、来源、顺序 | 按 task/episode/problem_version 定位的当前运行状态；旧快照最新真实进度 |
| `choose_next`、`_task_payload_for` | 决策时题面版本、必要 profile、选择事件 | decision/action 引用的共享不可变问题版本；旧 root/pre snapshot |
| `_recover_task_text`、检索文档 | 本次执行的题文版本及原文 | `task_text_digest` + `task_texts`，由执行/决策问题版本补足必要内容；旧快照兼容读取；不能用今天题面替换历史 |
| `build_prediction_context`、多候选调用 | 本轮 P/X/H/B、实际召回/反馈、候选统计、信息边界 | 一次决策一个有生命周期的运行输入；多候选共享，重试不重新召回 |
| 新/旧预测绑定、`service.py` 直接 SQL | 预测及执行各自身份版本、时间、config、scope、真实/假想来源 | 预测冻结条件 + action 的执行前条件；统一读取器回退旧 context/snapshot |
| 预算校准 | 预测时声明/确认状态，执行或 episode 真实声明及测量 | 预测公共冻结条件和既有真实预算账本；旧 snapshot.budget_state |
| `knowledge.py`、知识目标 | 候选对应 cell 统计、distinct tasks 数、命中知识的历史值 | 候选定向统计摘要、冻结 knowledge_targets、实际引用的内容版本；旧 coverage/cell_evidence |
| H+ 延迟验证 | 执行时间、目标知识当时可用性、采用声明、已有效果证据 | 执行/action 时间与目标相关的可用性凭据 + 独立 used_entries/reuse_trace；旧知识列表 |
| inspect/get/list/旧预测读取 | schema 版本、已存条件、来源与缺失 | 新统一兼容入口，显式报告 expired/missing/unsupported；旧对象保持可审计 |
| 归档、清理、版本对象回收 | 在线/归档/在途引用、待绑定/待检查/待验证状态 | 统一依赖枚举及运行状态生命周期；未解析旧引用保护，不猜测无依赖 |

## 5. 修改顺序及每步退出条件

### A1. 定义读取协议和兼容层，先固定基线

涉及 `core/schema.py`、`core/storage.py`、`world_model/state.py`、`context.py`、`contracts.py`。

- 建立一个统一条件解析入口，返回值附 `source`、`schema_version`、缺失原因和真实/假想状态。优先级：新冻结条件 → 旧上下文保存的值 → 原有快照；不得读当前银行来补历史。
- 新上下文结构建议显式版本为 `wm-context/4`，保留 `/3` 适配和原版本来源；未识别版本应可见报错，不能在列表中静默消失。不顺带提升风险事件、观测规则、校准版本。
- 将问题身份与完整任务 digest 分开保留，继续排除选择后写入的 model 对问题身份的影响；题文、数据、CIR 真变化仍要识别。
- 问题正文按实际问题版本共享，优先复用 `task_texts`，额外必要的结构化题面使用既有产物/内容版本接口承载；单独文件路径必须有内容版本及可恢复值，不能只有路径或 hash。
- 实际引用知识/执行证据按内容版本共享；不再为一次预测保存整库或整批匹配知识版本。预算 unknown、未声明、无预算须区分。
- 先建立固定夹具的召回身份、候选统计、知识目标、关键 provider 输入、校准聚合基线。

退出条件：旧 `/3` 上下文、旧执行记录和旧快照能通过统一读取器读取；结果、成本 mask、scope、方法 unknown 和引用语义未改变。

### A2. 先搬走后续依赖，再切换新写入

涉及 `api.py`、`actions.py`、`service.py`、`budget.py`、`planner.py`、`episode_closeout.py`、`maintenance_decision.py`、`knowledge.py`。

- 运行状态仅保存当前真实进度、问题版本引用、必要恢复字段；按 task/episode/problem_version 定位。action 保存必要执行前条件和进度增量，承担真实过程证据。
- 预测公共冻结条件包括身份、预算、必要 X、引用版本清单、模型/协议/提示版本、发送内容摘要及遗漏、真实/假想标记。多个候选引用同一份轻量条件；候选 method/config/scope/anchor 仍独立。
- 执行证据继续一条对应一次真实尝试，沿用 action 关联、task_text_digest、profile、method_planned/actual、usage、pending 和现有校验。没有观察到的 actual 不回填计划；used_entries 的缺失与 `[]` 继续区分。
- H+ 可用性与采用分别保存。对待验证目标在执行开始记录相关知识版本是否可用及依据，不能将“未展示/未采用”当成“不存在/无效”，也不能只保存召回命中而改变原可用性标准。执行发生时无法确认的内容明确未知；旧历史仍从旧快照读取。
- 所有消费者逐一切换，包含 `service.py` 的 SQL 和恢复辅助方法。预算比较不能因新快照不存在而静默不执行；假想来源也不能因找不到旧表而默认 real。
- 服务调用前持久化公共冻结依据；所有成功/失败路径使用同一保存入口原子写入预测及必要关联。网络调用期间不持有 SQLite 写事务。加入引用完整性检查和重试幂等性；明确外部调用已发生但未收到返回时 usage 未知。
- 避免循环嵌套现有 `transaction()`；需要组合写入时使用明确连接级操作或真正 savepoint，不把 `locked()` 当作跨多次 commit 的原子性保证。

退出条件：新任务从未创建完整快照，也能完成计划、选择、执行、失败暂存、绑定、题文恢复、预算校准和 H+ 验证；旧完整快照的回退仍能工作。

### A3. 临时输入、多候选共享与过期协议

涉及 `snapshot()` 所有调用点、`build_prediction_context`、`strategy_prediction.py`、`cli.py` 和 `storage.py`。

- 保留 `BeliefSnapshot` 作为内存组装/旧格式读取类型；新常规流程不向 `belief_snapshots` 追加完整对象，不再调用 `put_knowledge_version()` 冻结整批知识。
- 沿用 `prediction_contexts` 保存决策公共轻量冻结条件，另以最少量运行记录保存跨进程需要的定向输入。若需要新表，仅解决这份活跃输入的有限共享和恢复，不承载新的长期知识。
- 每次决策组装一次 P/X/H/B；仅携带候选相关统计、实际选择的记忆和反馈。删掉上下文的重复完整 snapshot 和无界支持 execution_ids/task_ids 清单，保留必要计数和被选证据版本。
- 运行输入按 decision/context 定位，具有 active、consumed/closed、expired 等明确状态。候选预测尚未完成、选择未消费、执行或恢复仍需要时保留；相关消费者完成后释放完整输入，只留审计 header 和轻量条件。未选择候选不无限期保留运行大对象。
- 中断恢复继续使用原输入，不读取今天的记忆替代；超时放弃的未完成运行状态明确 expired，要求开启新决策。过期后 inspect 可读轻量历史，复用预测输入接口明确报 context expired，不返回空上下文冒充成功。
- 对数量/字节设置运行期约束，排除无界积累的失败/中断对象。大题面的上限按内容版本共享评估，不承诺任意任务都是几 KB。
- `context --no-persist` 连同其内部快照路径真正不保留预测状态；显式 `snapshot` 命令说明临时有效期。旧 snapshot ID 的读接口不返回伪造完整历史。

退出条件：跨三个 CLI 进程的 plan-next → choose-next → execute 及中断重开可用；多候选共享 frozen 条件；活跃输入释放后绑定/评估仍可完成；固定候选/命中材料时加入无关策略不使持久 payload 线性增长。

### B. 校准诊断投影，单独报告模型输入变化

涉及 `episode_closeout.py`、`context.py:provider_view`、`strategy_prediction.py`。

- 单条评估的 stored/derived 明细与 published 摘要保持独立审计；上下文和模型只取适用聚合、修正计数/原因摘要、相关 pairs/H+。
- 删除模型输入中的完整 `validity_corrections`，按同一 model/metric/unit/scope/规则版本维持分组与过滤。摘要计数须说明统计总体，不能误称全部属于当前候选。
- 保留 sample 门槛、分字段 eligibility、late check 和 cost backfill；不合并不同观测版本、不把缺失标签变成 zero。
- 沿用完整请求体积检查，在最终装配/裁剪后记录实际发送内容的 hash、遗漏和大小诊断；上下文片段和完整请求两种口径均保留。
- A 的等价性在 B 之前独立验证；B 明确改变模型输入，token、时延、效果单独报告。若 A 中候选定向投影也改变实际发送语义，必须另列输入变化，不能藏在“纯存储解耦”中。

退出条件：所有 provider 输入不含修正明细；适用校准统计不变、修正摘要正确，超预算仍有真实诊断与遗漏记录。

### C. 固定记忆测试采用逐题隔离副本

优先复用现有 `--home` 和临时目录机制，不引入新的实验平台：

- 建库结束，以 SQLite backup API 保存一致性基库，固定已发布 calibration/pairs、策略知识及其索引。不能在 WAL 活跃时只复制单个 `.db` 文件。
- 每个测试题从同一基库创建独立可写 home，并复制/重建匹配的固定索引；题 A 的 record/check/closeout 只落 A 的副本，题 B 从基库重新创建，不复用 A home。
- 同题重试继续同一 home，允许自己的真实进度、预算、pending、日志和本题记录；实验结果另行汇总，不能回灌基库。
- 对比题 B 的召回/统计/已发布反馈与题 A 运行前基准，验证索引同步、consulted 更新和校准重发布均不跨题。

退出条件：题 A 的结果不改变题 B 可见记忆；题 A 内部重试仍继承自己的进度。提供精确运行命令和副本创建工具，而非声称已有 freeze CLI。

### D. 显式旧库维护，最后交付

建议新增 `maintain-snapshots` CLI（拟议命令，当前不存在），默认只读预览；`--apply` 与 `--backup` 显式执行，物理回收另加 `--vacuum` 或独立步骤。

- 预览使用真正只读 SQLite 连接，不能先构造会初始化 schema/切换 WAL 的普通 `Store`。报告可清理行数、payload 字节、每类保护引用、缺失字段和不能清理原因。
- 依赖范围覆盖 action pre/post、预测 input/context、活跃决策、待绑定、pending 执行、late check、H+、旧兼容读取和归档引用。无法解析的旧引用默认保护。
- 清理是显式内容迁移：在副本上提取可验证的必要冻结条件及内容版本，验证新来源可读，再删除对应无业务依赖的大对象。不会在普通开库/预测时重写历史。
- apply 前用 SQLite 一致性 backup 保存可恢复备份，检查备份完整性；持锁重新检查预览条件，原子提交受检的迁移/删除；重跑幂等。
- 被 abandoned/expired 的运行输入可以释放，但状态及已发生执行的必要事实不能丢失。共享内容回收枚举在线和归档存活引用，不因某一库释放就删仍被别处引用的对象。
- 分别报告 JSON payload 减少、SQLite freelist/page 变化、DB/WAL 实际文件变化；VACUUM 单独执行并验证，不将停写等同磁盘缩小。
- 真主库 apply 由用户显式发起；本轮后续实现验收也仅使用临时库和副本。

退出条件：preview 不改历史内容；apply 有备份、引用保护、失败恢复及幂等测试。未知旧依赖不会被硬删。

## 6. 验收矩阵及文档

| 验收组 | 必测行为 | 对应现有回归 |
| --- | --- | --- |
| 存储增长 | 无新完整快照、无重复 snapshot；固定候选和命中后增加无关策略/记录；大题面共享；运行状态数量/字节随消费释放 | `test_prediction_context`、`test_world_model`；新增专门生命周期/增长夹具 |
| 输入冻结与等价 | 多候选相同 frozen 条件；修改银行/预算/X 后原值不变；A 关键 provider 输入/目标/统计/召回一致 | `test_strategy_outcome`、`test_world_model_m2/m3/m6` |
| 事实链 | 不创建新完整快照仍可绑定/校准；问题变化与写 model 区分；假想不能绑定真实；失败、未选、未绑定、pending、usage 补录诚实 | `test_linkage_attribution`、`test_budget_stop`、`test_r23_budget_labels`、`test_method_evidence` |
| 兼容与跨进程 | `/3`、旧快照、新旧混合库；旧结果/成本/mask/校验不变；三进程执行、中断、过期、关联写入故障注入 | `test_end_to_end_cli`、`test_schema`；构造旧格式 fixture |
| 校准和 H+ | B 去掉明细但计数/聚合正确；预算变化不可比；可用/采用分开；晚到检查、未知、窗口不变 | `test_episode_closeout`、`test_calibration_v2`、`test_wm_calibration_loop`、`test_hplus`、`test_capability_evolution` |
| 清理与固定记忆 | 只读 preview；副本 apply/backup/幂等/保护；A→B 无跨题适应，同题重试有进度 | `test_evidence_window`、`test_cleanup_audit`；新增固定记忆/清理 fixture |
| 性能与文档 | 不引入预测时全历史扫描；保留 r22/r24 索引优化；全部相关示例和全量回归 | `test_r22_publish_performance`、`test_r24_archive_scan`、文档检查及示例 |

提示词中的 `test_legacy_upgrade.py` 本地只有缓存而没有 `.py` 源文件，不能据此报告已运行。旧格式兼容需要显式补可执行 fixture。

文档修改清单：主 `SKILL.md` 仅改快照职责、正常入口和维护入口；协议/lifecycle/兼容细节写入 `references/prediction_context.md`、`world_model_contract.md`、`strategy_outcome.md`、`episode_closeout.md` 和 `commands.md`；更新相关 examples 和中英文 README。README 目前仍写 `wm-context/2`，应与实际支持版本同步。

已有测试中直接断言 pre/post 完整快照存在的部分，改为检查条件、进度和关联行为；不能通过保留旧写入来满足旧实现细节断言。旧快照兼容测试则保留原格式样本。

当前可运行的核验命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests/harness -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python references/examples/_check_docs.py
```

正式实施时依次提交 A1 → A2/A3 → B → C → D；A2 与 A3 作为依赖闭环验收后才视为“停止长期快照”完成。每步记录固定夹具行数、payload 字节、写入增量与实际请求体积；完整快照回收与物理空间回收分别列账。

## 7. 本轮核验结果与后续执行条件

- 文档 flags/imports/result keys/links 检查通过；13 个示例被检查为存在，这不等于逐个运行全部示例。
- 沙箱内全量：1,678 tests，131.962s，2 failures / 9 errors。9 个 errors 均在本地 HTTP stub 创建 socket 时遭 `PermissionError: Operation not permitted`，不能视为业务回归。
- 2 个 failures 是 `test_world_model_m3.TestReviewFixes` 的单候选/最后候选墙钟超时断言；单独在沙箱内复跑 2 项时为 1 failure / 1 pass。它们依赖 0.01/0.08 秒墙钟预算，在当前环境有不稳定表现；建议改为可控时钟验证预算边界，不能直接放宽行为断言。
- 沙箱外全量复核：1,678 tests，115.367s，**OK**，退出码 0。源代码未修改，9 项 HTTP stub errors 和 2 项超时 failures 均未在此次复核出现。保留两次执行结果，不以最终通过抹去环境差异。
- 核验日志：`/tmp/or-harness-audit-tests.log`（受限首轮）、`/tmp/or-harness-audit-tests-unrestricted.log`（全量复核）、`/tmp/or-harness-audit-timeout-tests.log`（两项定向复跑）、`/tmp/or-harness-audit-docs.log`（文档检查）。
- Git 元数据可用后，先确认实际 `world-model-integration` HEAD 与已有改动，再建立隔离分支；远端连接不作为本地实现前提。当前不伪造分支或 commit，不自动推送。
- 实施前保留基库一致性备份和源码指纹。未执行旧库维护前，旧数据仍可由原代码读取；生成新格式后，回退代码须保留新格式读取能力或回滚到备份的独立 home，不能让旧代码静默忽略新事实。

完成定义：先证明新流程不再长期追加完整快照且后续事实链独立可用，再证明旧库直接读取和恢复正常，最后提供维护与固定测试命令。真实库大小、模型 token/时延/质量变化只有实测后才能报告。
