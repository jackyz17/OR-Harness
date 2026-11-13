# H+ 二值建库开关验收

日期：2026-11-13。讨论基线为 `00e19df`，实际接入基线为上一轮归纳/记忆改造后的
`main@cdf29f6834806a7adb6c0a38513ba7ae472ee513`。

已按要求执行 `git status --short` 和 `git rev-parse --short HEAD`。本环境 `.git`
为只读占位目录，二者均返回 `not a git repository`。通过 GitHub Git 数据接口核对
远程 HEAD，并对照修改前快照核验了 182 个本地受跟踪文件，全部与基线 blob 相同。
12 个原有本地独有文件保留，不纳入提交。

完整实测、版本检查、实际方法/采用记录、工件摘要和真实请求片段保存在
[hplus_binary.json](hplus_binary.json)。操作与外层交接说明见
[hplus.md](../../references/hplus.md)。

## 实际接入

`capability_gain.decision` 在同一次 `predict-strategy` 调用中给出 JSON 数值 0/1。
框架在预测记录的 `trace.model_info.hplus` 冻结权威值、原假设、具体核验和输入引用。
字符串、布尔、越界及非有限值按 0 调度，分别保留非法/缺失诊断；历史三态不转换。
有效 1 的说明或核验不完整也仍为 1。它不参与 G/C/R 策略效用。

候选被选中时返回核验交接，执行保存对应预测 ID 与核验计划。任务收尾先登记
既有规则试用和真实执行事实，再于证据淘汰前为绑定执行的 1 建立持久待办。
每个 episode × prediction_id 只有一个待办；失败尝试也调度，未执行预测保持未观测。
0 的执行、检查和采用事实照常保存，正式取材/写入均受门控，且允许它作为其他
正信号的对照案例。后置 allow/skip/pending 不能覆盖已冻结的二值值。

正常编排仍由外层 agent 推理并提交方法。`close-episode.induction_schedule`
给出确定的待办和命令；`induction-status` 提供恢复入口。`process_induction_jobs`
接入现有外层归纳回调，一次取材、一次提交短规则或空审查。框架校验同一个待办，
事务提交规则/回执，重启复用材料、动作和回执；未完成待办不能冻结策略库。

回执中的真实归纳动作与求解动作区分，自动绑定原预测及实际提交的规则版本。
新增、修订、补证据、已有覆盖、假设被推翻、材料不足均可记录；空审查不写规则。
来源题/已展示材料与未来独立目标分开。实质新版本还须通过原有独立迁移、边界和
必要回归才形成确认正例；充分且有来源的无增量比较形成反例，其余保持未知。
原假设兑现与实际知识增量分开记录。成本优势仍须同题、同验收和同计量范围对照。

## 正常调用

```bash
orx --home BANK --induction-mode hplus --world-model URL::MODEL \
  predict-strategy --task task.json --episode E --candidate candidate.json
orx --home BANK execute --task task.json --episode E --prediction P --code solve.py --workspace ws
orx --home BANK check-task EX --check '{"reference_objective":90,"reference_source":"bench_declared"}'
orx --home BANK record --from-staged EX
orx --home BANK close-episode --task T --episode E --terminal completed
orx --home BANK induction-status --task T --episode E
orx --home BANK induction-material --task T --induction-id J
orx --home BANK induce --induction-id J --review '{"outcome":"insufficient_material","hypothesis_outcome":"unknown","conclusion":"已核对实际方法和检查，尚无独立支持。"}' --note '完成本轮审查，不新增条目'
```

`P`、`EX`、`J` 使用前序返回的真实 ID。短规则通过同一个 `J` 的 `--relation`
提交；同编号修订继续使用 `entry_id`、`operation=revise` 和完整 when/do/expect。
外层 Python runner 调用 `process_induction_jobs(reviewer, task_id=T, episode_id=E)`，
回调返回普通 `induce` 参数。无需额外 H+ 预测或人工重复绑定。

## 端点规则实测

初始旧规则只适用最大化、正 X 系数时取上端点。正信号触发对最小化的同编号修订。

| 环节 | 真实执行/检查 | 结果 |
|---|---|---:|
| 旧来源题 | max 2X+5Y，X+Y≤1000，X≥2Y+300 | 2699 |
| 正信号修订来源 | 同一参数最小化 2X−5Y，取下端点 | 367 |
| 独立迁移 | min 2X−5Y，X+Y≤641，X≥2Y+130 | 90 |
| 同题错误方法对照 | 上述独立题实际取上端点 | 92，checker failed |
| 非单调边界 | min (X−3)²+Y，X+Y≤12，X≥2Y | 0，实际停用降维 |

修订产生同一条目版本 2，旧成功案例有明确回归记录；旧版本在候验期间保持生效，
独立迁移及边界通过后切换到版本 2。回传包括原预测、真实归纳动作、采用冻结版本和
独立题检查。已用于提炼的原题/同底题重试不会被统计为独立验证。
规则正确性依据有来源的脚本审阅、执行工件和原题 checker 声明范围；通用 checker
的 unchecked 项保留，未把一次通过当成自动自然语言证明。

另有回归验证：失败且说明/核验缺失的 1 仍产生待办，并能以材料不足空审查完成；
重复收尾、提交及中断恢复只有一份持久归纳动作；0 在正式取材前跳过且写入受阻，
但可进入另一正信号的材料；策略切换按各自绑定的预测和执行归属。

## 下一次请求

已从实际 H+ 分支的 `covered-min` 预测保存输入中导出请求片段，JSON 附件含完整
字段和 request digest。其关键内容为：

```json
{
  "hplus_convention": {
    "version": "wm-hplus-binary/1",
    "field": "capability_gain.decision",
    "values": [0, 1],
    "missing_or_invalid_schedule": 0
  },
  "prediction_context": {
    "task_id": "covered-min",
    "hplus_feedback": {
      "feedback_version": "wm-hplus-binary/1",
      "settled_examples": [{
        "decision": 1,
        "label": 1,
        "verification_state": "independently_validated_increment",
        "rule_versions": [{"entry_id": "1", "version": 2, "operation": "revise"}]
      }]
    }
  }
}
```

上面为实发请求的字段摘录，省略同包的其他案例/引用。模型看到的是预测时点前的
训练结算事实，按已知题族、粗结构和机制相关性先选取，再受既有反馈/上下文预算
约束。原上下文的反馈快照不会因后续独立验证而改变。未结案预测不会作为已确认先验。
原核验缺少完成回执时仍明确标 unknown，后续规则独立验证不会倒写该状态。

## 三组建库与冻结对照

同一初始库、同一任务流、同一替身预测器/归纳回调和验证标准，分别建库；建完
冻结，再为每道共同测试题从对应冻结库创建独立副本。随机组按种子匹配 H+ 的
实际归纳次数，随机选择不读取结果。

第一组故意构造一个正信号重复机会和一个漏掉的 0，验证负例与漏检审查路径：

| 分支 | 触发/回执 | 空审查 | 独立验证规则增量 | 确认的 0 漏检机会 | 冻结测试通过 |
|---|---:|---:|---:|---:|---:|
| H+ | 1 / 1 | 1 | 0 | 0 | 1 / 1 |
| NoH+ | 5 / 5 | 4 | 1 | 1 | 1 / 1 |
| 等预算随机 | 1 / 1 | 0 | 1 | 1 | 1 / 1 |

第二组将最小化修订设为正信号，验证 H+ 正常产出：

| 分支 | 触发/回执 | 空审查 | 独立验证规则增量 | 冻结测试通过 |
|---|---:|---:|---:|---:|
| H+ | 2 / 2 | 1 | 1 | 1 / 1 |
| NoH+ | 5 / 5 | 4 | 1 | 1 / 1 |
| 等预算随机 | 2 / 2 | 1 | 1 | 1 / 1 |

各分支共同冻结测试的整数目标值为 283，规则内容保持冻结。NoH+ 的 0 审查读取
预测当时的规则快照；补测证明后来写入的规则不会进入该次基线。归纳墙钟用真实
计时器测量，逐轮和总量见 JSON；宿主 tokens/tool_calls 没有实测来源，保持未知。
这些数字来自脚本化的判别/归纳器，验证的是流程和计数，不是实际模型优劣。

```bash
PYTHONPATH=src python -m or_harness.experiments.hplus --output /tmp/hplus-new-run --seed 7
PYTHONPATH=src python -m or_harness.experiments.hplus --output /tmp/hplus-positive-run --positive-min
```

正式实验复用 `run_comparison`，注入真实 provider、候选提出/执行和归纳回调。
最小 runner 每题一次预测尝试；已有重试编排在真实 episode 结束后复用同一个
`process_induction_jobs`。后续应以同口径实测确认判别力、漏检率和冻结测试效果。

## 测试与改动位置

- 定向回归：**295 项通过，0 失败、0 错误**。
- `tests/harness` 全量：**1847 项通过，0 失败、0 错误，169.396 秒**。
- 覆盖旧协议读取、Q/G/C/R 求解决策、收尾校准、输入快照、规则版本及正常 CLI/API。

```bash
PYTHONPATH=src:tests/harness python -m unittest \
  test_hplus_binary test_hplus_experiments test_capability_trace test_hplus \
  test_capability_evolution test_r14_induction test_r17_hplus_followup \
  test_wm_calibration_loop test_world_model_contract
PYTHONPATH=src:tests/harness python -m unittest discover -s tests/harness
```

实现：`api.py`、`cli.py`；新增 `world_model/hplus.py`、`experiments/hplus.py`；
修改 `world_model/{contracts,strategy_prediction,trace_archive,maintenance_decision,episode_closeout,maintenance,induction_gate,induction_material}.py`
及 `strategy/rule_versions.py`。

测试：新增 `test_hplus_binary.py`、`test_hplus_experiments.py`；调整
`test_capability_evolution.py`、`test_capability_trace.py`、`test_hplus.py`、
`test_r14_induction.py`、`test_r17_hplus_followup.py`、`test_rule_versions.py`、
`test_wm_calibration_loop.py`。旧独立离线维护测试明确使用 manual 流程，新 H+ 正式
流程另测双入口门控；没有跳过或关闭测试。

文档：`SKILL.md`、`README.md`、`references/{commands,concepts,episode_closeout,induction,prediction_context,strategy_outcome,world_model_contract}.md`，
新增 `references/hplus.md`、本报告及 JSON。完整文件清单也在 JSON 中。

仍需仓库外承担的是实际 agent 的求解实现、必要语义审阅和归纳推理，以及宿主用量
上报。可验证的交接条件是正信号待办 ID、真实方法/检查来源、对应回执和已提交版本。
缺少这些事实时保持未知；框架不会代替 agent 推导规则，也不会据次数或预测值认定收益。
