# GenRec 进度维护

- 最后更新：2026-04-20
- 维护目的：用一个稳定文件追踪当前各数据集的实验阶段、已经跑过的线、试过的 idea，以及下一步。

## 1. 更新规则

- 这份文档优先记录“实验线/数据集级别”的进展，不记录每条命令。
- 如果旧周报正文和较新的导出表不一致，优先使用最新的本地 summary CSV / `checkpoint-*/metrics.json`。
- 需要展开细节时，链接到对应 dated note，而不是把整篇周报重复抄过来。

## 2. 全局快照

| 数据集 | 当前阶段 | 当前主结论 |
| --- | --- | --- |
| `Instruments` | 主线最完整，SFT 与多轮 RL 比较已完成 | clean fixed-hint 仍是最平衡主线；dynamic family 里 `gather-fix` 是默认基线，`max1` 是 early-stop 候选 |
| `Games` | 已跑通 `index -> preprocess -> SFT -> RL` 全流程 | `fixed-hint` 当前最平衡，`rule_only` 继续体现 top-10 与 coverage 的交换 |
| `Arts` | 只有基础脚本支持，尚无正式实验记录 | 可以开始，但当前还没有已落地的 SFT/RL 结果和文档线 |

## 3. Instruments

### 当前阶段

`Instruments-grec` 仍是当前最主要的研究驱动数据集。这里已经完成：

- SFT 基线
- reward form 消融
- dynamic vs fixed hint 主线对比
- corrected fixed-hint 及 `sid-only` 变体
- CE / `hintce-2`
- dynamic `max1` 消融

### 当前最好用的基线与候选

按当前本地导出表，最常引用的几条线如下：

| 实验线 | Best checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 | 当前读法 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `rule_only rerun` | `checkpoint-2997` | `0.0960` | `0.1179` | `0.1070` | `0.1681` | top-10 极值线，但 coverage 明显偏低 |
| `dynamic gather-fix` | `checkpoint-2997` | `0.0936` | `0.1169` | `0.1083` | `0.1855` | 当前 dynamic family 默认对比线 |
| corrected `fixed taskfix sid-only` | `checkpoint-2652` | `0.0945` | `0.1205` | `0.1103` | `0.1935` | 当前 clean fixed-hint 参考上界 |
| corrected `fixed taskfix` | `checkpoint-2997` | `0.0931` | `0.1189` | `0.1094` | `0.1941` | coverage 很强，且有早期 spike |
| old `fixed mixed-single` | `checkpoint-3326` | `0.0953` | `0.1193` | `0.1114` | `0.1938` | 历史强参考，但带 legacy bug，不能当 clean 方法定义 |

补充两个 peak coverage 读数：

- corrected `fixed taskfix` 的 `HR@50` 峰值是 `0.1962 @ checkpoint-666`
- old `fixed mixed-single` 的 `HR@50` 峰值是 `0.1957 @ checkpoint-2331`

### 已经尝试过的 idea

#### 1. Reward form 消融

已跑过：

- baseline mixed GRPO
- `rule_only`
- `prefix only`
- `prefix seq only`
- `prefix token only`
- `prefix tokenadv raw`
- `prefix tokenadv totalnorm`
- `prefix tokenadv totalnorm errtok`

当前结论：

- `rule_only` 仍然是最强的无 hint top-10 baseline。
- prefix 类 reward 没有替代 exact reward。
- token-level prefix reward 对 normalization 很敏感，`raw` 明显不稳，`totalnorm` 才能恢复成可比结果。

#### 2. Hint scaffold 主线

已跑过：

- `dynamic hint sid-only`
- `dynamic hint gather-fix`
- `ranking dynamic cascade`
- old `fixed hint mixed-single`
- corrected `fixed hint taskfix`
- corrected `fixed hint taskfix sid-only`

当前结论：

- fixed family 整体比 dynamic family 更稳、更接近当前 top-right trade-off。
- `dynamic gather-fix` 是当前应该保留的 canonical dynamic baseline。
- corrected `fixed taskfix sid-only` 已经足够接近 old fixed 的现象学上界，适合作为后续 clean fixed 基线。

#### 3. Fixed-hint 后续变体

已跑过：

- `fixed taskfix + CE`
- `fixed taskfix + CE (hintce-2)`
- `fixed taskfix + CE (hintce-3)`
- `dynamic max1`
- `fixed dual-task`

当前结论：

- full `+CE` 更像把训练轨迹拉平的正则项。
- `hintce-2` 是当前最强的 CE 变体：
  `checkpoint-2664 / NDCG@10=0.0931 / NDCG@50=0.1102 / HR@50=0.1951`
- `max1` 是当前最值得继续解释的 shallow-budget dynamic：
  best 点在 `checkpoint-1332 / NDCG@10=0.0934 / HR@50=0.1905`
  但长跑到后段会回落，因此更像 early-stop 候选，不是新的默认长跑线。
- `hintce-2` 的训练日志拆分已经补过一轮：
  当前 `weighted_hint_ce_loss` 大体稳定在 `~1e-3`，前期相对 `RL base` 略重，但主 spike 仍然主要由 `KL` 驱动，而不是 CE 本身。
- 仓库现在已经支持一条 dual-task filtered setting：
  `task1_sid_sft + task5_title_desc2sid` 参与 train，`task4_hisTitle2sid` 被移除，eval 仍只看 `task1_sid_sft`；dynamic / fixed 两个 launcher 都已落地，其中 `dynamic dual-task` 现在已经同步到 `10` 个 checkpoint（`checkpoint-302` 到 `checkpoint-3012`），而 `fixed dual-task` 也已经开始同步到 `checkpoint-302/604/906`。
- mixed-task `single-hint` setting 已经补到中后段 checkpoint：
  训练仍保留三任务，但只对 `task1_sid_sft` 注入 fixed hint，`task4/task5` 强制 zero-hint；
  当前本地已同步到完整 `checkpoint-3326`，best readout 保持在
  `NDCG@10=0.0948 / HR@10=0.1180 / NDCG@50=0.1116 / HR@50=0.1958`。
- `single-hint mixed` 不再只是 early-window strong candidate；
  它现在在 `checkpoint-2664` 已经同时压过 corrected `fixed taskfix sid-only` 的 `NDCG@10` 和 `HR@50`，而且到完整尾点 `checkpoint-3326` 仍维持 `HR@50=0.1951`，因此已经是需要认真对待的主候选。
- `dynamic dual-task` 现在更像一条“已经可比较、但还没赢 baseline”的完整首轮轨迹：
  raw full-trace best 点仍是 `checkpoint-1510 / epoch≈1.003 / NDCG@10=0.0930 / HR@50=0.1885`；
  最新同步到 `checkpoint-3012` 也还没有把它抬回 best window。
- `fixed dual-task` 当前只是 first-look：
  `checkpoint-302/604/906` 已同步，best 点是
  `checkpoint-906 / NDCG@10=0.0910 / HR@50=0.1795`；
  同一 aligned slot 下它的 top-10 略高于 dynamic dual-task 的 `checkpoint-906`，但 coverage 仍明显更弱。
- `hintce-3` 当前只同步了 `checkpoint-333/666` 两个 early readout：
  `checkpoint-666 / NDCG@10=0.0898 / HR@50=0.1924`；
  目前还没看到能替代 `hintce-2` 的证据，更像一个需要继续补长的倍率试探。

### 当前最值得继续做的事

- 把 `UFT-style hint curriculum` 落到 corrected `fixed taskfix sid-only` 上。
- 对 `hintce-2` 做更细的 task-level 和训练日志分析，确认为什么 balanced 优势出现在中后段而不是最终点。
- 把 `fixed dual-task` 和 `hintce-3` 继续补到更多 checkpoints，再判断它们到底是 early noise 还是有稳定信号。
- 继续观察 `dynamic dual-task` 能否从当前 `1510` 左右的 best 点往上推，还是 `3012` 之后仍然维持“中段最佳、尾段回落”的形态。
- 解释 `single-hint mixed` 为什么在完整 `2.0` epoch 尾段维持高位平台，但没有超过 `checkpoint-2664` 的 top-10 峰值。
- 对 `max1` 补回 train-time 日志和 diagnostics，尤其是 `selected_hint_depth_mean`、`selected_depth_1_frac`、`mean_length`。
- 在主线对比里优先保留 4 条：`rule_only rerun`、`dynamic gather-fix`、corrected `fixed taskfix sid-only`、old `fixed mixed-single`（仅作历史参考）。

### 主要参考文档

- [2026-03-18-genrec-main-results-weekly.md](2026-03-18-genrec-main-results-weekly.md)
- [2026-04-02-genrec-results-since-2026-03-19-epoch-report.md](2026-04-02-genrec-results-since-2026-03-19-epoch-report.md)
- [2026-04-11-genrec-instruments-rl-variant-comparison.md](2026-04-11-genrec-instruments-rl-variant-comparison.md)
- [2026-04-11-instruments-rl-next-ideas.md](2026-04-11-instruments-rl-next-ideas.md)
- [2026-04-16-instruments-dynamic-hint-max1-ablation.md](2026-04-16-instruments-dynamic-hint-max1-ablation.md)
- [2026-04-19-instruments-dual-task-single-hint-tracking.md](2026-04-19-instruments-dual-task-single-hint-tracking.md)

## 4. Games

### 当前阶段

`Games-grec` 已经从 `index -> preprocess -> SFT -> RL` 全流程跑通，不再是“只有数据和脚本，没有结果”的状态。

目前已完成：

- 单数据集 `Games` semantic index 训练与导出
- `Games_grec` 数据构建
- `Games-grec` SFT checkpoint 对比
- 三条 RL 主线：
  `rule_only`、`dynamic-hint`、`fixed-hint`

### 当前主结果

| 实验线 | Best checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 | 当前读法 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `SFT` | `checkpoint-768` | `0.0433` | `0.0804` | `0.0691` | `0.1998` | 当前 SFT 最优点 |
| `rule_only` | `checkpoint-8752` | `0.0467` | `0.0825` | `0.0683` | `0.1815` | top-10 提升最明显，但 coverage 掉得最多 |
| `dynamic-hint` | `checkpoint-4380` | `0.0464` | `0.0823` | `0.0716` | `0.1980` | 基本救回 coverage，但 top-10 仍略弱于 fixed |
| `fixed-hint` | `checkpoint-8752` | `0.0480` | `0.0857` | `0.0723` | `0.1972` | 当前最平衡主线 |

补充一个很关键的 coverage 峰值：

- `fixed-hint` 在 `checkpoint-3504` 达到 `HR@50=0.2024`

### 已经尝试过的 idea

#### 1. 单数据集 semantic index

已完成：

- `qwen3-embedding-4B`
- `rq4(cb256-256-256-256)`

当前结论：

- index 已足够进入下游 SFT/RL。
- 第 0 层 codebook 利用率偏低、后三层打满，这后面值得单独分析，但不阻碍主线实验推进。

#### 2. SFT checkpoint 选择

已完成：

- `checkpoint-128` 到 `checkpoint-1024` 的统一评测对比。

当前结论：

- 纯 SFT best point 是 `checkpoint-768`
- 继续接 RL 时，`checkpoint-896` 作为初始化点更稳妥

#### 3. RL 三线

已完成：

- `rule_only`
- `dynamic-hint`
- `fixed-hint`

当前结论：

- `Games` 已经开始复现 `Instruments` 上的主故事。
- `rule_only` 继续体现“top-10 换 coverage”。
- `dynamic-hint` 说明 hint scaffold 对 coverage 有帮助，但还没超过 `fixed-hint`。
- `fixed-hint` 当前是最值得继续加算力和复现的线。

### 当前最值得继续做的事

- 继续把 `fixed-hint` 作为 `Games` 主讲结果线维护下去。
- 补更多 task-level 或训练日志证据，确认 `Games` 上 fixed / dynamic / no-hint 的差异是否与 `Instruments` 同机制。
- 保持把新增结果续写进 [2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md](2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md)，不要拆成近重复文档。

### 主要参考文档

- [2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md](2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md)
- [2026-04-17-genrec-main-results-weekly.md](2026-04-17-genrec-main-results-weekly.md)

## 5. Arts

### 当前阶段

截至本次整理，`Arts` 还没有进入“有正式实验记录”的阶段。

当前仓库快照里能确认的状态是：

- `evaluate_all_checkpoints.sh` 和 `evaluate_all_checkpoints_sidecar.py` 已经预留了 `Arts` 路由。
- `scripts/index/` 下有多份 `Instruments-Arts-Games` 联合 index 训练/导出脚本。
- 但 `docs/` 里没有 `Arts` 顶层实验笔记。
- `results/` 里没有 `*Arts*` 结果目录。
- `examples/train_full/` 里也还没有 `Arts` 单独的 SFT 配置目录。

### 已经尝试过的 idea

目前能确认的只包括基础设施层面的准备：

- `Arts` 已被纳入统一评测脚本的默认数据路由。
- 仓库里已经有 `Instruments-Arts-Games` 联合 index 的脚本模板。

还不能确认的内容：

- 没有看到已完成的 `Arts` 单数据集 index
- 没有看到已完成的 `Arts-grec` preprocess / SFT / RL
- 没有看到配套的 dated note

### 当前最值得继续做的事

- 先决定 `Arts` 要走单数据集路线，还是先走 `Instruments-Arts-Games` 联合 index 路线。
- 一旦开始第一条正式实验线，就补齐三件东西：
  `examples/train_full/Arts/...`、`results/Arts...`、`docs/YYYY-MM-DD-arts-*.md`
- 在正式 run 开始前，这一节可以先保留为“准备状态”记录，不要假装已经有结果。
