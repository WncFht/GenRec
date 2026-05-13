# GenRec 周报（2026-04-17，Games + Instruments）

- 记录日期：2026-04-17
- 汇报范围：`Games-grec` 新主线结果，以及 `Instruments-grec` 本周继续推进的 `UFT` / `CE` / `CE-2` / `max1` 几条尝试
- 统一设置：`Qwen2.5-3B-Instruct` RL，`qwen4B-4-256` semantic ID，`split=grec`
- 相关详细记录：
  - [2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md)
  - [2026-04-11-genrec-instruments-rl-variant-comparison.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-11-genrec-instruments-rl-variant-comparison.md)
  - [2026-04-16-instruments-dynamic-hint-max1-ablation.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-16-instruments-dynamic-hint-max1-ablation.md)
  - [2026-04-11-instruments-rl-next-ideas.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-11-instruments-rl-next-ideas.md)
- 结果来源：
  - `Games` 口径优先使用 [`games_grec_rl_variant_best_summary.csv`](/Users/fanghaotian/Desktop/src/GenRec/docs/assets/2026-04-01-games-grec-qwen4b-4-256/games_grec_rl_variant_best_summary.csv)
  - `Instruments CE` 口径优先使用 [`fixed_hint_ce_best_summary.csv`](/Users/fanghaotian/Desktop/src/GenRec/docs/assets/2026-04-11-genrec-instruments-rl-variant-comparison/fixed_hint_ce_best_summary.csv)
  - `Instruments max1` 口径优先使用 [`max1_ablation_best_summary.csv`](/Users/fanghaotian/Desktop/src/GenRec/docs/assets/2026-04-16-instruments-dynamic-hint-max1-ablation/max1_ablation_best_summary.csv)

说明：

- `2026-04-11` 和 `2026-04-16` 两篇正文里的 `CE` / `hintce-2` / `max1` 描述，部分还停在更早的 checkpoint 同步状态。
- 这份周报里涉及 `hintce-2` 和 `max1` 的数字，按当前仓库里已经更新的本地导出表写，不按旧正文里“只到 `1332` / `1665`”的口径写。

## 1. 本周结论

1. `Games-grec` 这周最大的进展是：从 `index -> preprocess -> SFT -> RL` 的全流程已经真正跑通，而且 RL 三条主线已经给出了稳定可比的结果。当前排序和 `Instruments` 很像，按 `best NDCG@10 checkpoint` 看是 `fixed-hint > rule_only ≳ dynamic-hint > SFT`。
2. `Games` 上当前最值得讲的 headline 仍然是 `fixed-hint`。它在 `checkpoint-7884` 达到 `NDCG@10=0.0479`、`HR@10=0.0857`、`NDCG@50=0.0721`、`HR@50=0.1970`，而且更早的 `checkpoint-3504` 还打到了当前最高 `HR@50=0.2024`。
3. `Instruments` 这周继续拆三条线看：
   - `UFT-style` 方向被明确成后续主线之一，重点不是搬整套框架，而是借它的 `hint curriculum + hinted-token SFT loss`。
   - `CE` 现在更像把 `fixed-hint` 训练轨迹拉平的正则项。
   - `hintce-2` 和 `max1` 都给出了比上周更值得继续追的信号，但都还没到可以直接升成默认线的程度。
4. 在 `Instruments` 的 clean fixed family 里，当前本地最新导出表显示：
   - non-CE `fixed taskfix` 的 best 点仍是 `checkpoint-2997 / NDCG@10=0.0931 / HR@50=0.1941`
   - full `+CE` 的 best 点是 `checkpoint-2664 / NDCG@10=0.0915 / HR@50=0.1944`
   - `hintce-2` 现在也已经补到 `checkpoint-2664`，best 点同样达到 `NDCG@10=0.0931`，同时 `HR@50=0.1951`
   也就是说，它已经不再只是 `1332` 处的 early-run readout。
5. 在 dynamic family 里，`max1` 仍然是当前最有希望的 shallow-budget 变体。它的 best 点在 `checkpoint-1332`，达到 `NDCG@10=0.0934`、`HR@50=0.1905`，相对 `rule_only` 只少 `0.0026` 的 `NDCG@10`，但多了 `0.0224` 的 `HR@50`。不过当前本地表也已经补到 `checkpoint-2997`，后段确实有回落，所以更准确的说法仍然是“前中期 trade-off 很强”，不是“已经赢过 gather-fix”。

## 2. Games：全流程已跑通，fixed-hint 先成为主讲结果

### 2.1 流程状态

- `Games` 单数据集的 `qwen3-embedding-4B + rq4(cb256x4)` index 已训练并导出，最终 collision rate 为 `0.0075873`。
- 基于该 index 的 `Games_grec` 数据已构建完成。
- SFT 当前最佳点按 `NDCG@10` 看是 `checkpoint-768 / 0.0433`，而 RL 默认起点使用更稳妥的 `checkpoint-896`。
- 本地 `results/` 里三条 RL 主线都已经同步到完整 2-epoch 轨迹。

### 2.2 当前主结果

| Variant | Best checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| `SFT` | `checkpoint-768` | `0.0433` | `0.0804` | `0.0691` | `0.1998` |
| `rule_only` | `checkpoint-8752` | `0.0467` | `0.0825` | `0.0683` | `0.1815` |
| `dynamic-hint` | `checkpoint-4380` | `0.0464` | `0.0823` | `0.0716` | `0.1980` |
| `fixed-hint` | `checkpoint-7884` | `0.0479` | `0.0857` | `0.0721` | `0.1970` |

再补一个这周汇报里最好直接说的 coverage 点：

- `fixed-hint` 在 `checkpoint-3504` 的 `HR@50=0.2024`，是当前 `Games` 主线里最高的 coverage 峰值。

### 2.3 这组结果最重要的读法

1. `Games` 现在已经不是“pipeline 搭好了但 RL 还没结果”的状态，而是开始出现和 `Instruments` 相同的结构性趋势。
2. `rule_only` 依然最像“拿 top-10 换 coverage”的线。它把 `NDCG@10` 从 `0.0433` 提到 `0.0467`，但 `HR@50` 掉到了 `0.1815`，明显低于 SFT 的 `0.1998`。
3. `dynamic-hint` 在 `Games` 上也表现出熟悉的特征：coverage 能被救回来，但 top-10 还没超过 `fixed-hint`。
4. `fixed-hint` 目前仍然是 `Games` 最平衡的一条线。它不只是拿到了最高的 `NDCG@10`，也基本保住了 `HR@50`，而且给出了全表最高 coverage 峰值。

一句话概括 `Games` 本周结果：

- `Games` 已经开始复现 `Instruments` 上“hint scaffold 比 plain exact reward 更平衡”的主故事，而 `fixed-hint` 是当前最值得继续加算力和复现的方向。

## 3. Instruments：继续把 hint 线拆细

### 3.1 UFT：本周先把“该借什么”想清楚

这周 `UFT` 还不是一条已经完成评测的结果线，更准确地说，它是被明确成了下一阶段最值得落地的 fixed-hint 变体。

当前最重要的判断是：

1. 我们真正想借的不是 `UFT` 的整套训练框架，而是它的两件事：
   - 训练过程中逐步减少 hint 暴露的 `hint curriculum`
   - 只作用在 hinted token 上的轻量 `SFT / CE` 项
2. 这条线最合理的第一落点不是 dynamic，而是 clean fixed baseline：
   - `fixedhint-taskfix-b16-sid-only`
3. 它想回答的问题不是“UFT 论文有没有赢”，而是：
   - fixed-hint 的 coverage 收益能不能在 hint 逐步退场后保住
   - old fixed 那种“更浅但更稳”的优势，能不能用显式 curriculum 复现，而不是继续依赖 bug

所以周报里更合适的写法是：

- 本周已经把 `UFT-style hint curriculum` 明确成 `Instruments` fixed-hint 的下一条主尝试，后续会先落在 corrected `fixedhint-taskfix-b16-sid-only` 上。

### 3.2 CE / CE-2：现在不只是“加不加 CE”，而是开始分化出两种作用

按当前本地最新导出表，`fixed taskfix` family 的三条线可以写成：

| Variant | Best checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 | Peak HR@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| non-CE `fixed taskfix` | `checkpoint-2997` | `0.0931` | `0.1189` | `0.1094` | `0.1941` | `0.1962 @ checkpoint-666` |
| full `fixed taskfix + CE` | `checkpoint-2664` | `0.0915` | `0.1180` | `0.1081` | `0.1944` | `0.1947 @ checkpoint-3326` |
| `fixed taskfix + CE (hintce-2)` | `checkpoint-2664` | `0.0931` | `0.1168` | `0.1102` | `0.1951` | `0.1954 @ checkpoint-1665` |

这张表现在可以支持三个判断：

1. full `+CE` 仍然更像“把轨迹拉平”的项，而不是 headline 指标增强项。它的 top-10 低于 non-CE，但 late coverage 平台更稳。
2. `hintce-2` 现在已经不该再写成“只在 `333/666/999/1332` 上起量更快”的早期变体。当前本地表已经补到了 `checkpoint-2664`，它在这个点上：
   - `NDCG@10` 追平 non-CE (`0.0931`)
   - `NDCG@50` 还更高 (`0.1102` vs `0.1094`)
   - `HR@50` 也更高 (`0.1951` vs `0.1941`)
3. 但 `hintce-2` 还没有完整跑到和 non-CE / full `+CE` 一样的 `3326` horizon，所以当前最稳妥的定位仍然是：
   - 它已经从“值得继续的 early-train 变体”升级成“当前最有希望的 CE-2 候选”
   - 但还没有足够证据直接替代 full `+CE` 或 non-CE 成为默认 fixed 线

### 3.3 max1：dynamic family 里最有希望的 shallow-budget 候选

`max1` 这周的核心信息是：它确实不是简单退化成 `rule_only`。

按当前本地最新导出表，它和几条主参考线的 best-point 对比如下：

| Variant | Best checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| `rule_only` | `checkpoint-2997` | `0.0960` | `0.1179` | `0.1070` | `0.1681` |
| `dynamic gather-fix` | `checkpoint-2997` | `0.0936` | `0.1169` | `0.1083` | `0.1855` |
| `dynamic max1` | `checkpoint-1332` | `0.0934` | `0.1158` | `0.1095` | `0.1905` |
| corrected `fixed taskfix sid-only` | `checkpoint-2652` | `0.0945` | `0.1205` | `0.1103` | `0.1935` |

这组数更适合支持下面三个结论：

1. 相比 `rule_only`，`max1` 明确买回了 coverage，而且买得不贵：
   - `NDCG@10` 只低 `0.0026`
   - `HR@50` 高 `0.0224`
2. 相比 `dynamic gather-fix`，`max1` 在 best 点上几乎没有损失 top-10，但把 coverage 又往 fixed family 推了一点：
   - `NDCG@10 -0.0002`
   - `HR@50 +0.0050`
3. 但它还不是新的 fixed 替代品。corrected `fixed taskfix sid-only` 仍然整体更强，而且当前本地表已经把 `max1` 补到 `checkpoint-2997`，后段确实回落到了 `NDCG@10=0.0920 / HR@50=0.1783`。所以现在最好把它写成：
   - “当前 dynamic family 里最接近 fixed trade-off 的候选”
   - 而不是“已经稳定超过 gather-fix 或 fixed”
