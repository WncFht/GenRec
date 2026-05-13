# GenRec 周报（2026-04-29，Instruments Hint / CE 后续）

- 记录日期：2026-04-29
- 汇报范围：`Instruments-grec` 本周新同步和新整理的 `CE scaling`、`sid-hint-only / single-hint mixed`、`dual-task`、`hint-token SFT`
- 研究稿来源：[`docs/research/instruments_rl_research_report.tex`](research/instruments_rl_research_report.tex)，研究稿日期为 2026-04-26
- 统一设置：`Qwen2.5-3B-Instruct` RL，`qwen4B-4-256` semantic ID，`split=grec`
- 结果来源：本地同步的 `results/*/checkpoint-*/metrics.json`，以及研究稿拆分后的 `docs/research/sections/*.tex`

说明：

- 这份周报不再重复 2026-04-17 周报里已经讲过的 `Games` 全流程、早期 `UFT` 讨论和 `max1` first-look 内容。
- 旧的 `rule-only`、clean fixed baseline 只作为坐标保留；本周正文聚焦新增结果和新设定。

## 1. 本周结论

1. `Instruments-grec` 的主故事没有被推翻：plain `rule-only` 仍是 top-10 最强的无 hint baseline，但 `HR@50=0.1681` 的 coverage 损失太明显；真正值得继续互相比的是 clean fixed、`single-hint mixed`、`hintce-3`、`hintce-4` 这一组更平衡的候选。
2. `CE` 本周已经从“试一个辅助项”推进成完整的倍率/实现对照。`hintce-3` 在 `checkpoint-1665` 达到 `NDCG@10=0.0945 / HR@50=0.1985`，是当前最强 coverage 峰值；`hintce-4` 把 best `NDCG@10` 推到 `0.0953 @ checkpoint-2997`，说明更高倍率没有直接失效，而是换成了更偏后段 top-10 的曲线形状。
3. `sid-hint-only mixed` 现在可以更准确地写成 `single-hint mixed`：训练仍是 mixed 三任务，但只对 `task1_sid_sft` 开 hint。fixed 版本已经是当前最值得认真对待的新候选，best 点为 `checkpoint-2664 / NDCG@10=0.0948 / HR@50=0.1958`。
4. `dynamic single-hint mixed` 和 `dynamic hint + CE(0.005)` 都补到了更长轨迹，但结果没有改写 dynamic family 排序。二者现在更像机制边界的负结果，而不是需要优先追加算力的新主线。
5. `single-hint mixed + CE(0.005)` 已经补齐完整轨迹，但没有把两个 parent line 的优势叠加起来：best 点 `NDCG@10=0.0943 / HR@50=0.1919`，低于 plain `single-hint mixed` 的 coverage，也低于 `hintce-3` 的 coverage。
6. `hint-token SFT` 是本周最清楚的新方法定义：把 RL loss、reward、advantage、KL/ref model 全部拿掉，只在 beam-selected oracle hint token 上做 masked SFT。当前 fixed / dynamic 两个 launcher 和轻量单测已经就绪，但还不是结果线。

## 2. 本周实验口径

本周只讲 `Instruments-grec`，并按 best `NDCG@10` 对齐 checkpoint，再看对应 `HR@50`。少量旧线只用于定位新结果：

| Variant | Best checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 | 读法 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `SFT495` | `checkpoint-495` | `0.0823` | `0.1094` | `0.0985` | `0.1844` | 共同参考点 |
| `rule-only rerun` | `checkpoint-2997` | `0.0960` | `0.1179` | `0.1070` | `0.1681` | top-10 极值线，coverage 最差 |
| `single-hint mixed` | `checkpoint-2664` | `0.0948` | `0.1180` | `0.1116` | `0.1958` | 当前最强 partial-hint 候选 |
| `fixed taskfix sid-only` | `checkpoint-2652` | `0.0945` | `0.1205` | `0.1103` | `0.1935` | clean fixed reference |
| `fixed CE coef=0.005` | `checkpoint-1665` | `0.0945` | `0.1180` | `0.1118` | `0.1985` | `hintce-3`，coverage 峰值最强 |
| `fixed CE coef=0.01` | `checkpoint-2997` | `0.0953` | `0.1204` | `0.1113` | `0.1943` | `hintce-4`，后段 top-10 最强 |

这张表的重点不是选一个单独冠军，而是确认当前前沿集合：`single-hint mixed`、clean fixed、`hintce-3`、`hintce-4` 已经挤到相近区域，各自代表不同 trade-off。

## 3. CE 调参

这周 CE family 最重要的变化是：`hintce-2/3/4` 和 dynamic CE 都已经能放进同一张图里读，不再只是早期 checkpoint 的片段。

| Variant | Best checkpoint | Epoch | NDCG@10 | HR@50 | 读法 |
| --- | --- | ---: | ---: | ---: | --- |
| `fixed taskfix` | `checkpoint-2997` | `1.802` | `0.0931` | `0.1941` | non-CE 参考线 |
| `hintce` | `checkpoint-2664` | `1.602` | `0.0915` | `0.1944` | batch-mean CE |
| `hintce-2` | `checkpoint-2664` | `1.602` | `0.0931` | `0.1951` | token-mean / 更接近 DAPO 口径 |
| `hintce-3` | `checkpoint-1665` | `1.001` | `0.0945` | `0.1985` | CE 倍率 `0.005`，coverage 峰值最强 |
| `hintce-4` | `checkpoint-2997` | `1.802` | `0.0953` | `0.1943` | CE 倍率 `0.01`，后段 top-10 追上历史上界 |
| `dynamic hint + CE(0.005)` | `checkpoint-1998` | `1.201` | `0.0917` | `0.1851` | full-trace dynamic CE，未超过 canonical dynamic |

本周更合适的读法是：

- `hintce-2` 是更稳的 token-mean / DAPO-compatible CE 候选。
- `hintce-3` 更像中段爆发出的高 coverage / 高 NDCG 候选。
- `hintce-4` 说明 `0.01` 倍率不是直接过强失效，而是把 top-10 往训练后段继续抬高；但它还没有超过 `hintce-3` 的 coverage 峰值。
- `dynamic hint + CE(0.005)` 已经不应继续写成“还要等后段”。它补到 `checkpoint-3326` 后仍没有超过 canonical `dynamic gather-fix`；peak `HR@50` 仍停在 `checkpoint-666 / 0.1910`，尾点回落到 `0.1839`。

## 4. Sid-Hint-Only / Single-Hint Mixed

这组实验容易混淆，所以周报里建议直接用下面的定义：

- `fixed single-hint mixed`：仍训练 mixed 三任务，只对 `task1_sid_sft` 注入 fixed hint；`task4/task5` 走 zero-hint。
- `dynamic single-hint mixed`：仍训练 mixed 三任务，只对 `task1_sid_sft` 开 dynamic cascade；其他任务强制 no-hint。

对应 launcher：

- `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-hint-only-mixed.sh`
- `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-hint-only-mixed.sh`

关键结果：

| Variant | Best checkpoint | Epoch | NDCG@10 | HR@50 | 读法 |
| --- | --- | ---: | ---: | ---: | --- |
| `fixed taskfix` | `checkpoint-2997` | `1.802` | `0.0931` | `0.1941` | clean fixed 参考 |
| `fixed taskfix sid-only` | `checkpoint-2652` | `2.000` | `0.0945` | `0.1935` | clean sid-only 上界 |
| `fixed single-hint mixed` | `checkpoint-2664` | `1.602` | `0.0948` | `0.1958` | 当前最强 partial-hint 候选 |
| `dynamic single-hint mixed` | `checkpoint-2331` | `1.402` | `0.0911` | `0.1823` | 已同步到 `2997`，仍弱于 dynamic baseline |

结论很直接：partial-hint 的主要故事目前发生在 fixed 版本上。`fixed single-hint mixed` 同时压过 clean `fixed taskfix sid-only` 的 `NDCG@10` 和 `HR@50`，说明“只在 task1 上给 hint、其他任务保留 no-hint 分布”可能比 full fixed 更接近当前想要的 trade-off。

相反，`dynamic single-hint mixed` 的 long-run 没有延续 early-window 信号。它的 peak `HR@50` 在 `checkpoint-1665` 只有 `0.1874`，尾点回落到 `NDCG@10=0.0905 / HR@50=0.1802`；这条线现在更像 partial-hint 机制边界的对照项。

## 5. Single-Hint Mixed + CE(0.005)

这条线表面上是在做：

- `single-hint mixed`
- 加上 `hintce-3` 同款 CE 倍率 `0.005`

但研究稿里已经把关键 caveat 写清楚了：它不是一个干净的 additive ablation。因为 `single-hint` launcher 同时把 `FIXED_HINT_TASK_NAMES`、`ANALYSIS_TASK_NAMES`、`EVAL_TASK_NAMES` 固定成 `task1_sid_sft`，所以新线实际上是：

- `task1-only fixed-hint`
- `task1-only prompt-side CE`
- `task4/task5` 不仅没有 hint，也没有 hint CE 梯度

完整轨迹结果如下：

| Variant | Best checkpoint | Epoch | NDCG@10 | HR@50 | 读法 |
| --- | --- | ---: | ---: | ---: | --- |
| `fixed single-hint mixed` | `checkpoint-2664` | `1.602` | `0.0948` | `0.1958` | parent line 1 |
| `fixed CE coef=0.005` / `hintce-3` | `checkpoint-1665` | `1.001` | `0.0945` | `0.1985` | parent line 2 |
| `single-hint mixed + CE(0.005)` | `checkpoint-2664` | `1.602` | `0.0943` | `0.1919` | 完整轨迹后确认的负结果 |

所以这条线现在不应写成“还没等到后段翻盘”。它已经补齐到 `checkpoint-3326`，但 best 点仍低于两个 parent line 的关键优势，尤其 coverage 明显更弱。更合理的结论是：partial-hint task gating 和 prompt-side CE 在当前实现下不会自动叠加成更好的 balanced candidate。

## 6. Dual-Task 与 Hint-Token SFT

### 6.1 Dual-Task

dual-task family 只保留 `task1 + task5`，移除 `task4_hisTitle2sid`。它现在已经有 fixed / dynamic 两条可比轨迹：

| Variant | Best checkpoint | Epoch / aligned epoch | NDCG@10 | HR@50 | 读法 |
| --- | --- | --- | ---: | ---: | --- |
| `dynamic gather-fix` | `checkpoint-2997` | `1.802` | `0.0936` | `0.1855` | canonical dynamic |
| `dynamic dual-task` | `checkpoint-1510` | `1.003 / 1.861` | `0.0930` | `0.1885` | 可比，但未超过 dynamic 主线 |
| `fixed dual-task` | `checkpoint-2718` | `1.805 / 1.972` | `0.0939` | `0.1897` | 强于 dynamic dual-task，但弱于 clean fixed |
| `fixed taskfix sid-only` | `checkpoint-2652` | `2.000` | `0.0945` | `0.1935` | clean fixed reference |

当前定位：dual-task 是值得保留的 filtered setting，但还没有改写主线排序。fixed dual-task 明显比 dynamic dual-task 更像可追候选，不过离 clean fixed / single-hint mixed 还有差距。

### 6.2 Hint-Token SFT

本周新增的 `hint-token SFT` 不是 RL 结果，而是一个更干净的新训练定义：

- 不算 reward。
- 不做 advantage 标准化。
- 不用 old/ref log-prob。
- 不加 KL penalty。
- 不用 completion-token policy gradient。
- 只在 prompt suffix 的 oracle hint token 上做 masked token-mean CE。

对应入口：

- Trainer：`hint_sft_trainer.py`
- fixed launcher：`hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-hint-token-sft-fixed.sh`
- dynamic launcher：`hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-hint-token-sft-dynamic.sh`
- 单测：`tests/test_hint_sft_trainer.py`

这两个 launcher 默认都跑 Instruments mixed 三任务，不启用 `sid-hint-only` 或 task filter。默认设置已经改成更符合纯 SFT 的口径：

- `learning_rate=3.0e-4`
- `hint_sft_loss_coef=1.0`
- 默认 `5` epoch

当前验证状态是：masked batch builder、fixed train step、dynamic depth selector 都已有轻量检查，两个 launcher 也跑过 `--dry-run`。下一步是等真实结果出来后，把 WandB / sidecar 指标接进现有 `docs/research/scripts/build_instruments_report_figures.py` 图表流程。

## 7. 下周建议

1. 优先解释 `single-hint mixed` 为什么能在完整 2 epoch 里维持 fixed family 前沿，而不是马上继续扩大量 dynamic single-hint。
2. 对 `hintce-3` 和 `hintce-4` 分开讲：前者是中段高 coverage 峰值，后者是后段 top-10 抬高；不要再用“CE 好/不好”一句话混在一起。
3. `single-hint mixed + CE(0.005)`、`dynamic hint + CE(0.005)`、`dynamic single-hint mixed` 都可以作为机制性负结果保留，但不建议优先追加算力。
4. `hint-token SFT` 是下一条最干净的新验证线。等 fixed / dynamic 两条真实 run 跑完后，再决定它是替代 CE 辅助项，还是只作为离线 hint imitation 的诊断工具。
