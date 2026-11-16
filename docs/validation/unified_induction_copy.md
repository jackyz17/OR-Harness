# 归纳文档审核稿

以下文案已接入 `SKILL.md` 与 `references/{induction,hplus}.md`。本文件单列供审核；命令中的 J、EX_A、EX_B、EX_BOUNDARY 必须使用框架返回的实际 job 和执行 ID。只有已执行、已派发的归纳待办才能提交正式审阅。

## SKILL.md：正常收尾与归纳

求解前，`plan-next` 在每个候选的同一次预测中冻结 H+。U 使用 G/C/R；H+ 独立决定本次归纳投入。实际执行后，读取 `close-episode` 的 `result.induction_schedule.pending_ids`。逐一取材并提交，确认 `induction-status.completed:true` 后进入下一题。执行失败、已有覆盖或证据不足都不取消冻结的正信号，可以提交有依据的空审阅回执。

预测时的 **H+=0 是跳过新归纳**，仍可记录执行、已有规则使用、检查和效用管理。下面的**学习标签 0** 来自一次真实审阅，两者含义不同。规则发布是条目的使用决定；学习标签取决于真实归纳结果及证据支持。

归纳正文继续写 `when / do / expect`。Agent 自主选择可比较案例、适用边界、检查和发布范围。已有执行加推导或小规模边界试验可以支持所声明范围内的发布，不要求先有第三道真实题。

### 有据的新增或修订

先执行 `orx induction-material --induction-id J`。随后同时提交短规则和学习结论：

```bash
orx induce --induction-id J \
  --relation '{
    "when":"Y可有限枚举，固定Y后整数X的非空可行域是有限区间，且目标对X单调。",
    "do":"跳过没有可行X的Y。对其余Y按目标方向选相应区间端点，比较所有Y。目标对X不单调时，完整枚举可行X。",
    "expect":"在所述条件下保留原问题的最优目标值，并减少对X的枚举。",
    "evidence":["EX_A","EX_B","EX_BOUNDARY"],
    "publication":{"state":"published","reason":"端点推导和记录的小规模枚举支持这一范围。","scope":["有限整数区间、固定Y时目标对X单调"]}
  }' \
  --review '{
    "outcome":"new_rule",
    "conclusion":"明确了端点选择方法和非单调目标下的回退条件。",
    "comparison":{
      "sufficient":true,"substantive":true,"basis":"derivation",
      "source":"实际A、B题的输入和枚举记录，以及小规模非单调边界试验。",
      "assessment":"两种目标方向都由端点推导解释，边界试验核对了完整枚举回退。",
      "evidence_refs":["EX_A","EX_B","EX_BOUNDARY"]
    }
  }'
```

`--relation` 保存规则、证据和发布选择；`--review` 保存真实学习结论。实际新版本与有据的实质增量可以得到学习标签 1。修订已有条目时在 relation 中增加实际 `entry_id` 和 `operation:"revise"`，保留该编号及版本链；outcome 可用 `revised`。发布本身不自动产生学习正例。

### 已覆盖或没有增量

可以完成空审阅，也可以在同一提交中给实际已有版本补证据。空审阅示例：

```bash
orx induce --induction-id J \
  --review '{
    "outcome":"already_covered",
    "conclusion":"实际执行采用的端点方向及回退条件已由当前规则版本覆盖，没有可复用的新增量。",
    "comparison":{
      "sufficient":true,"substantive":false,
      "source":"已对照当前规则版本、实际执行步骤及原题检查记录。",
      "assessment":"实际方法落在已有适用条件内，对照没有发现需要新增或修订的步骤或边界。",
      "evidence_refs":["EX_A","EX_B"]
    }
  }'
```

充分对照且引用有效检查事实时可得到学习标签 0；没有增量也可用 `no_increment`。需要补证据时加 `--relation '{"entry_id":"实际编号","operation":"add_evidence","evidence":["EX_A"]}'`，与上述 review 一起结清原 J。材料不足、仅未召回旧规则或只改写措辞不能自动充当充分对照。

### 证据不足

```bash
orx induce --induction-id J \
  --review '{
    "outcome":"insufficient_material",
    "conclusion":"已审阅实际输入与执行记录，但缺少约束覆盖检查，无法核对端点回退是否保持原题可行性；本轮不新增规则。"
  }'
```

该命令完成本轮审阅并保留回执，学习标签为未知。不能把未知写成无增量。后续有效补检、更正、引用失效按现有证据规则进入之后的反馈；已经发出的预测仍保留当时输入。

## references/induction.md：对应拟用段落

常规顺序为 `profile / recall → plan-next → execute --prediction → check-task → record → close-episode → 按冻结 H+ 接单`。一个或多个候选走同一评分入口，修复求解代码后重新提案和评分。取材从 J 的真实绑定执行开始；agent 决定对照、推导、边界检查与发布范围。提交时同时保留短规则与 review 的学习结论，按上面的新增/修订、已有覆盖和材料不足三种写法结清待办。同编号版本、证据引用和维护入口保持原义。

## references/hplus.md：对应拟用段落

H+ 是求解前的二值投入开关，与 U 无关。只对实际获准、绑定并执行的候选派单；未执行、影子评分或校验拒绝没有执行标签或归纳任务。H+=1 即使失败或材料不足也要完成一轮审阅。H+=0 跳过新归纳；已有条目的采用、检查和效用管理继续使用原入口。

有据的新版本及实质知识增量可得到学习标签 1；充分比较后确认已有覆盖或无可用增量可得到学习标签 0；材料不足、未审阅或唯一支持无效为未知。上述三类示例均使用真实派发的 J，不能把事前 H+=0 当作已审阅的学习负例。发布和学习标签分别记录，空审阅也必须保留结论和回执。


## 已执行的完整提交命令

以下 J 和三个 EX 来自本轮服务器临时库的实测记录，保留了实际提交的英文规则与 review 原文。此命令已通过 CLI 重放，返回同一归纳回执，不新增归纳动作。临时路径只在本次服务器验收中有效；新建库请用本次 `close-episode` 与 `execute` 返回的实际 ID。

```bash
PYTHONPATH=src python3 -m or_harness.cli --home /tmp/unified-endpoint-review induce --induction-id ij_687aaca0be5371420445828f \
  --relation '{
  "do": "Enumerate Y, derive both X endpoints, choose by coefficient sign and objective direction, and compare objectives. Disable endpoint reduction for nonmonotone objectives and enumerate all feasible X,Y.",
  "evidence": [
    "ex_3603365320e7",
    "ex_8affcfb0e36f",
    "ex_9ffa257cf24d"
  ],
  "expect": "Preserve the original integer optimum and all stated constraints.",
  "publication": {
    "reason": "At fixed Y a monotone objective attains the interval optimum at the selected endpoint; enumerations and the nonmonotone fallback were checked.",
    "scope": [
      "monotone integer-interval endpoint derivation",
      "A/B enumerations",
      "constructed nonmonotone fallback"
    ],
    "state": "published"
  },
  "when": "Nonnegative integer X,Y; at fixed Y the feasible X set is an interval and the objective is monotone in X."
}' \
  --review '{
  "comparison": {
    "assessment": "Derive the feasible X interval; monotonicity and objective direction choose its endpoint. The constructed nonmonotone case uses full enumeration.",
    "basis": "derivation",
    "evidence_refs": [
      "ex_3603365320e7",
      "ex_8affcfb0e36f",
      "ex_9ffa257cf24d"
    ],
    "source": "demo inducer compared captured integer inputs, endpoint algebra and actual enumerations",
    "substantive": true,
    "sufficient": true
  },
  "conclusion": "The direction-aware endpoint mechanism and its disable boundary are supported in the stated scope.",
  "outcome": "new_rule"
}'
```

实测：A=2699、B=90、非单调边界=0；在第三道真实 C 题前发布带范围的规则，之后 C=196。H+ 学习标签为 1；已有条目保持管理没有新增归纳动作。回执：`hplus_receipt:dafc1d15a125c1e73ff2b38d86b484094a165e2664ee700da0e6890f6c631038`。
