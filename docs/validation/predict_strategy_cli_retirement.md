# 独立预测 CLI 退役验收

日期：2026-11-17。接续统一 U 精简，核对基线为 `main@d430c99d6505ede0586c499e508f80e3ab2dcac6`。
当前目录没有可用的 Git 元数据；实际执行 `git status --short`、`git rev-parse --short HEAD` 均报告非 Git 仓库。
通过 GitHub ref 与上一轮提交的完整文件清单确认基线，修改前受跟踪文件未发生后续差异。12 份用户本地独有文件保持原样，不纳入提交。

## 改动与保留的行为

- 删除 `cli.py` 中独立预测子命令的注册及 `cmd_predict_strategy()`；没有保留别名或重定向。
- 保留 `ORHarness.predict_strategy_outcome()` 的签名和完整实现，修改前后 AST 一致。`plan-next` 继续调用它，每个有效候选一次预测，共享冻结上下文并计算 U。
- 单候选、多候选均读取 `result.selected_prediction_id`，直接交给 `execute --prediction P`。原有任务、配置、评分、选中许可与单次消费校验继续生效。
- 查看已存预测使用 `orx inspect --bank predictions --prediction P`，不触发模型调用。沿用原有结果结构：`result.prediction.kind` 表示记录类型，`result.prediction.prediction` 为预测正文。
- 更新操作文档、README、示例说明、结果键清单及文档检查注册。两份历史验收报告只迁移操作指引，历史实测数据、ID 与版本引用保持原样。
- 归纳的 `--relation`、`--review` 正文和 H+ 审阅语义均未修改。

正常操作：

```bash
# candidates.json 为含一个或多个候选的 JSON 列表。
orx --world-model URL::MODEL plan-next --task task.json --episode E --candidates candidates.json
# P = result.selected_prediction_id
orx inspect --bank predictions --prediction P
orx execute --task task.json --prediction P --code solve.py --workspace ws
orx check-task EX --check CHECK
orx record --from-staged EX
orx close-episode --task T --episode E --terminal completed
```

## 实测结果

| 验收 | 结果 |
|---|---|
| 隔离安装当前源码后调用真实 `orx --help` | 退出码 0；无旧子命令；包含 `plan-next`。安装采用本地 wheel、无网络依赖下载。 |
| 直接调用已移除的命令 | 退出码 2，argparse 报 `invalid choice`。负向回归仍保留旧名称作为测试输入。 |
| 单候选 HTTP 预测与读取 | 返回可执行 P/U，只调用 provider 一次；inspect 读回同一 P，没有新增调用。 |
| 多候选 HTTP 预测 | 两个候选调用两次；收到同一冻结上下文；选中原预测。 |
| 首提成本参照与执行上下文 | 普通 `plan-next` 保留传给模型的宿主、阶段、成本证据；允许并核实一条真实 `select_strategy` 决策动作，没有伪执行记录。 |
| 正常单候选/多候选端到端链路 | 原有 CLI 示例通过评分、执行、检查、记录、收尾、H+ 审阅与重试。各模式均执行两道题，分别调用 provider 2/4 次，执行复用原预测。 |
| 定向回归 | 172 项，35.552 秒，全部通过。 |
| `tests/harness` 全量 | 1,891 项，212.756 秒，全部通过。 |
| 文档检查 | 参数、命令、符号、链接、91 项结果键及 14 个示例文件检查通过。 |
| 语法检查 | `compileall` 通过。 |

实际运行命令：

```bash
PYTHONPATH=src:tests/harness python3 -m unittest -q \
  test_strategy_outcome.TestCliStrategyCommands test_c_signal_review \
  test_documented_result_keys test_end_to_end_cli test_candidate_normalization \
  test_command_surface test_snapshot_lifecycle
PYTHONPATH=src:tests/harness python3 -m unittest discover -s tests/harness -q
PYTHONPATH=src python3 references/examples/_check_docs.py
python3 -m compileall -q src tests/harness references/examples
```

本机 HTTP 替身 provider 仅验收调用链。上述结果不构成真实策略判别力的实验结论。

## 修改文件

实现与契约说明：

- `src/or_harness/cli.py`
- `src/or_harness/api.py`（执行拒绝信息的术语更新）
- `src/or_harness/world_model/contracts.py`（候选规整文档字符串）

测试：

- `tests/harness/test_strategy_outcome.py`
- `tests/harness/test_c_signal_review.py`
- `tests/harness/test_documented_result_keys.py`
- `tests/harness/test_end_to_end_cli.py`
- `tests/harness/test_world_model_m3.py`（测试说明）

操作文档、示例和检查清单：

- `SKILL.md`、`README.md`、`README_zh.md`
- `references/commands.md`、`references/strategy_outcome.md`
- `references/prediction_context.md`、`references/snapshot_lifecycle.md`
- `references/examples/linkage_attribution.py`（示例说明）
- `references/examples/_check_docs.py`、`references/examples/_result_keys.py`
- `docs/validation/hplus_binary.md`、`docs/validation/unified_solving.md`（操作指引）
- 本报告 `docs/validation/predict_strategy_cli_retirement.md`
