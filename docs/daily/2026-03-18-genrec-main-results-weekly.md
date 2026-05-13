# GenRec 本周主结果周报（Reward Form + Hint Ablation，2026-03-18）

- 首次生成: 2026-03-18
- 更新日期: 2026-03-19
- 汇报范围: `GenRec` 当前 `Instruments-grec` 主线 RL 实验
- 基线 SFT: `GenRec/results/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495`
- 统一设置: `Qwen2.5-3B-Instruct` RL, `qwen4B-4-256` semantic ID, `split=grec`
- 结果来源: `GenRec/results/*/checkpoint-*/metrics.json`
- 脚本来源: `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/`

## 1. 本周一句话结论

- 本周最强的 `NDCG@10` 结果来自 `rule_only rerun`，最佳点为 `checkpoint-2997`，`NDCG@10=0.0960`。
- 如果只看 top-10，`rule_only` 比当前 baseline mixed GRPO 还略强；但它和 baseline 一样，都存在 `HR@50` 低于 SFT 的问题。
- 本周最值得强调的新结果仍然是 `fixed hint + rule_only`，而且这条线又继续抬高了：
  - 按 `NDCG@10` 选 best checkpoint，最佳点更新为 `checkpoint-3326`
  - `NDCG@10=0.0953`
  - `HR@10=0.1193`
  - `NDCG@50=0.1114`
  - `HR@50=0.1938`
- 如果单独看 coverage 峰值，`fixed hint` 在 `checkpoint-2331` 仍然达到当前最高 `HR@50=0.1957`。
- 这说明固定 oracle hint scaffold 现在已经基本追平 baseline mixed GRPO 的 top-10 强度，同时继续保持更强的 top-50 覆盖指标，是当前最有希望缓解 `rule_only` 稀疏奖励副作用的方向。

## 2. 实验口径

本周主结果只统计已经有明确脚本和结果目录对应关系、且当前可以稳定解释的实验组。

- SFT 基线:
  - `Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495`
- Reward form ablation:
  - `baseline_grpo`
  - `rule_only`
  - `prefix_only`
  - `prefix_seq_only`
  - `prefix_token_only`
  - `prefix_tokenadv_raw`
  - `prefix_tokenadv_totalnorm`
  - `prefix_tokenadv_totalnorm_errtok`
- Hint ablation:
  - `dynamic_hint_rule_only`
  - `fixed_hint_rule_only`

说明:

- `prefix-seq-only` 采用本周修正后的 `fixbool-rerun` 结果，不使用更早的异常 run。
- `rule_only` 采用 `rerun-quietlog` 结果，不使用更早的旧目录。
- `dynamic hint` 结果已补到 `checkpoint-2664`，本文按当前最佳点 `checkpoint-1665` 统计。
- 文中“相对提升(%)”统一按 `(当前值 - 对照值) / 对照值 * 100%` 计算。

## 3. SFT 基线

| Stage | Checkpoint | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| SFT baseline | `checkpoint-495` | 0.0823 | 0.1094 | 0.0985 | 0.1844 |

## 4. Reward Form Ablation

### 4.1 结果表

| Variant | NDCG@10 | HR@10 | NDCG@50 | HR@50 | Delta NDCG@10 vs SFT | Rel. NDCG@10 vs SFT | Delta HR@10 vs SFT | Rel. HR@10 vs SFT | Delta NDCG@50 vs SFT | Rel. NDCG@50 vs SFT | Delta HR@50 vs SFT | Rel. HR@50 vs SFT | Best Ckpt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline GRPO | 0.0952 | 0.1145 | 0.1071 | 0.1696 | +0.0129 | +15.7% | +0.0051 | +4.7% | +0.0086 | +8.7% | -0.0148 | -8.0% | `checkpoint-1665` |
| rule only | 0.0960 | 0.1179 | 0.1070 | 0.1681 | +0.0137 | +16.6% | +0.0085 | +7.8% | +0.0085 | +8.6% | -0.0163 | -8.8% | `checkpoint-2997` |
| prefix only | 0.0862 | 0.1049 | 0.0982 | 0.1612 | +0.0039 | +4.7% | -0.0045 | -4.1% | -0.0003 | -0.3% | -0.0232 | -12.6% | `checkpoint-1332` |
| prefix seq only | 0.0857 | 0.1046 | 0.0985 | 0.1636 | +0.0034 | +4.1% | -0.0048 | -4.4% | +0.0000 | +0.0% | -0.0208 | -11.3% | `checkpoint-1332` |
| prefix token only | 0.0815 | 0.1005 | 0.0958 | 0.1668 | -0.0008 | -1.0% | -0.0089 | -8.1% | -0.0027 | -2.7% | -0.0176 | -9.5% | `checkpoint-1665` |
| prefix tokenadv raw | 0.0476 | 0.0597 | 0.0564 | 0.0991 | -0.0347 | -42.2% | -0.0497 | -45.4% | -0.0421 | -42.7% | -0.0853 | -46.3% | `checkpoint-999` |
| prefix tokenadv totalnorm | 0.0906 | 0.1125 | 0.1054 | 0.1809 | +0.0083 | +10.1% | +0.0031 | +2.8% | +0.0069 | +7.0% | -0.0035 | -1.9% | `checkpoint-3326` |
| prefix tokenadv totalnorm errtok | 0.0796 | 0.0968 | 0.0914 | 0.1513 | -0.0027 | -3.3% | -0.0126 | -11.5% | -0.0071 | -7.2% | -0.0331 | -18.0% | `checkpoint-999` |

### 4.2 分析

#### 结论 1: 当前最强 top-10 指标仍然是 `rule_only`

- `rule_only` 的最佳 `NDCG@10=0.0960`，相对 SFT 提升 `+16.6%`，略高于 baseline mixed GRPO 的 `0.0952`（相对 SFT `+15.7%`）。
- `rule_only` 的 `HR@10=0.1179`，相对 SFT 提升 `+7.8%`，也高于 baseline mixed GRPO 的 `0.1145`（相对 SFT `+4.7%`）。
- 这说明在当前 `Instruments-grec` 设定下，最稀疏的 exact-match 奖励并没有拖垮 top-10 排序，反而给了最强的 top-10 结果。

#### 结论 2: prefix reward 不能直接替代 exact reward

- `prefix_only` 和 `prefix_seq_only` 虽然比 SFT 只有 `+4.7%` / `+4.1%` 的 `NDCG@10` 相对增益，但明显弱于 `rule_only` 和 baseline mixed GRPO。
- 二者的共同特征是：
  - `NDCG@10` 只到 `0.0857~0.0862`
  - `HR@10` 反而低于 SFT
- 这说明“只奖励前缀对了多少”不足以支撑最终 next-item 排序质量，至少在当前设定下，它更像辅助信号，不像主导信号。

#### 结论 3: token-level prefix advantage 对归一化非常敏感

- `prefix_tokenadv_raw` 的结果最差，`NDCG@10` 只有 `0.0476`，相对 SFT 下降 `-42.2%`，说明原始 token-level 方案非常不稳定。
- 一旦加入 `totalnorm`，结果立刻恢复到 `0.0906`，相对 SFT 变成 `+10.1%`，比 raw 方案高出 `+0.0430`。
- 这说明本周 reward 线里最明确的工程结论是：
  - token-level prefix reward 不是不能用
  - 但如果不做合适的 total-token normalization，训练会明显崩掉

#### 结论 4: 当前 `errtok penalty` 版本没有带来额外收益

- `prefix_tokenadv_totalnorm_errtok` 相比 `prefix_tokenadv_totalnorm` 明显变差：
  - `NDCG@10`: `0.0796` vs `0.0906`
  - `HR@10`: `0.0968` vs `0.1125`
- 至少在目前这版实现和当前数据设置下，把 NDCG 惩罚只打到 error token 上，并没有比普通 totalnorm 更好。

#### 结论 5: top-10 变强，不等于 top-50 一定同步改善

- baseline mixed GRPO 和 `rule_only` 虽然 top-10 都强，但二者 `HR@50` 都低于 SFT：
  - baseline mixed GRPO: `0.1696`，相对 SFT `-0.0148`
  - rule_only: `0.1681`，相对 SFT `-0.0163`
- 这延续了我们上周已经观察到的结构性现象：
  - RL 会优先把 top-10 打磨得更强
  - 但 top-50 覆盖面容易收缩

## 5. Hint Ablation（在 `rule_only` 主线下）

### 5.1 结果表

| Variant | NDCG@10 | HR@10 | NDCG@50 | HR@50 | Delta NDCG@10 vs SFT | Rel. NDCG@10 vs SFT | Delta HR@10 vs SFT | Rel. HR@10 vs SFT | Delta NDCG@50 vs SFT | Rel. NDCG@50 vs SFT | Delta HR@50 vs SFT | Rel. HR@50 vs SFT | Best Ckpt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| plain rule only | 0.0960 | 0.1179 | 0.1070 | 0.1681 | +0.0137 | +16.6% | +0.0085 | +7.8% | +0.0085 | +8.6% | -0.0163 | -8.8% | `checkpoint-2997` |
| dynamic hint rule only | 0.0919 | 0.1160 | 0.1080 | 0.1905 | +0.0096 | +11.7% | +0.0066 | +6.0% | +0.0095 | +9.6% | +0.0061 | +3.3% | `checkpoint-1665` |
| fixed hint rule only | 0.0953 | 0.1193 | 0.1114 | 0.1938 | +0.0130 | +15.8% | +0.0099 | +9.0% | +0.0129 | +13.1% | +0.0094 | +5.1% | `checkpoint-3326` |

### 5.2 分析

#### 结论 1: `dynamic hint` 已经不只是早期趋势信号

- `dynamic hint` 现在已经补到 `checkpoint-2664`，按 `NDCG@10` 最佳点看，`checkpoint-1665` 达到：
  - `NDCG@10=0.0919`
  - `HR@10=0.1160`
  - `NDCG@50=0.1080`
  - `HR@50=0.1905`
- 相对 SFT，它已经实现：
  - `NDCG@10 +11.7%`
  - `HR@10 +6.0%`
  - `NDCG@50 +9.6%`
  - `HR@50 +3.3%`
- 它仍然没追上 plain `rule_only` 的最强 top-10，但 `HR@50` 相比 plain `rule_only` 仍高出 `+0.0224`，相对提升约 `+13.3%`。
- 这说明在线 cascade hint 至少已经释放出一个很有价值的信号：
  - 它不只是“先把覆盖面救回来”
  - 它在 top-10 和 top-50 两侧都已经有了稳定增益
  - 但当前整体水平仍低于 `fixed hint`

#### 结论 2: `fixed hint` 是本周最值得强调的新结果

- `fixed hint rule only` 按 `NDCG@10` 选 best checkpoint，现已更新为 `checkpoint-3326`：
  - `NDCG@10=0.0953`
  - `HR@10=0.1193`
  - `NDCG@50=0.1114`
  - `HR@50=0.1938`
- 如果单看覆盖峰值，`checkpoint-2331` 的 `HR@50=0.1957` 仍是当前最高点。
- 相比 plain `rule_only`：
  - `NDCG@10` 只低 `0.0007`
  - 相对下降仅 `-0.7%`
  - `HR@10` 高 `+0.0014`
  - 相对提升 `+1.2%`
  - `NDCG@50` 高 `+0.0044`
  - 相对提升 `+4.1%`
  - `HR@50` 高 `+0.0257`
  - 相对提升 `+15.3%`
- 相比 baseline mixed GRPO：
  - `NDCG@10` 高 `+0.0001`
  - 相对提升 `+0.1%`
  - `HR@10` 高 `+0.0048`
  - 相对提升 `+4.2%`
  - `NDCG@50` 高 `+0.0043`
  - 相对提升 `+4.0%`
  - `HR@50` 高 `+0.0242`
  - 相对提升 `+14.3%`

可以把这条结果概括成一句话：

- `fixed hint` 现在已经基本追平甚至略超 baseline mixed GRPO，同时显著改善了 coverage 指标，是当前最平衡的一条 RL 变体。

#### 结论 3: `fixed hint` 说明 “给 scaffold” 比 “改 reward 形状” 更有希望

本周 reward ablation 说明：

- 单独改 prefix reward 形状，效果很不稳定
- 稍有不合适的归一化，结果就会明显退化

而 `fixed hint` 的结果说明：

- 与其在 reward 里反复设计更细的 token-level shaping
- 不如直接给模型一个更稳定的 oracle scaffold，缓解 exact reward 的稀疏性

这也是为什么本周最值得在组会上强调的不是 `prefix_tokenadv_totalnorm`，而是 `fixed hint rule only`。

## 6. 训练轨迹观察

### 6.1 baseline mixed GRPO

- 最佳点出现在较早阶段 `checkpoint-1665`
- 后续虽然仍维持较高 `NDCG@10`，但 `HR@50` 明显回落到 `0.16x`
- 说明它的主收益集中在早期，后期更容易收缩覆盖

### 6.2 plain `rule_only`

- 从 `checkpoint-333` 到 `checkpoint-2997` 基本持续提升
- `NDCG@10` 从 `0.0902` 稳步涨到 `0.0960`
- 说明这条线虽然奖励稀疏，但训练趋势比我们原先担心的更稳定

### 6.3 `dynamic hint rule only`

- 从 `checkpoint-333` 到 `checkpoint-1665`，四个主指标都在持续改善
- 最佳 `NDCG@10` 出现在 `checkpoint-1665`
- 后续 `1998 -> 2664` 有一定回落，说明这条线当前更像“中期最好、后期略退”
- 但即便如此，它的 `HR@50` 仍长期维持在高于 SFT 的区间

### 6.4 `fixed hint rule only`

- 从 `checkpoint-333` 到 `checkpoint-3326`，`NDCG@10` 基本持续抬升
- 按 `NDCG@10` 统计，最佳点更新到 `checkpoint-3326`
- 按 coverage 峰值看：
  - `HR@50` 在 `checkpoint-2331` 达到 `0.1957`
  - `NDCG@50` 在 `checkpoint-2997` 达到 `0.1117`
- 这说明 fixed hint 不只是某个偶然尖峰，而是一条在 top-10 与 top-50 两侧都更稳定的训练轨迹

## 7. 本周最推荐的汇报口径

如果这周汇报只想突出三点，建议直接讲下面三句：

1. 当前 `Instruments-grec` 上，best `NDCG@10` 仍然来自 `rule_only`，达到 `0.0960`。
2. 单独做 prefix reward 不能替代 exact reward，token-level reward 还高度依赖 normalization。
3. 本周最有潜力的新方向是 `fixed hint + rule_only`，它按 best `NDCG@10` 选点已经达到 `0.0953`，并且训练过程中 `HR@50` 峰值达到当前最好 (`0.1957`)。

## 8. 后续建议

- 下一周如果继续按“主结果周报”口径推进，最值得优先补的不是更多 prefix 变体，而是：
  - 补齐 `dynamic hint` 的完整训练轨迹
  - 继续验证 `fixed hint` 的稳定性与可复现性
- 如果要继续做 reward 线，最合理的对照主轴应收缩为：
  - `baseline mixed GRPO`
  - `rule_only`
  - `prefix_tokenadv_totalnorm`
  - `fixed_hint_rule_only`

这四条线已经足够代表本周主要发现。

## 附录

### A.1 基线结果目录

| Item | 结果目录 |
| --- | --- |
| SFT baseline | `Instruments-grec-sft-qwen4B-4-256-dsz0` |

### A.2 Reward Form Ablation 结果目录

| Variant | 结果目录 |
| --- | --- |
| baseline GRPO | `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| rule only | `Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| prefix only | `Instruments-grec-grpo-prefixonly-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| prefix seq only | `Instruments-grec-grpo-prefix-seq-only-fixbool-rerun-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| prefix token only | `Instruments-grec-grpo-prefix-token-only-totalnorm-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| prefix tokenadv raw | `Instruments-grec-grpo-prefix-tokenadv-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| prefix tokenadv totalnorm | `Instruments-grec-grpo-prefix-tokenadv-totalnorm-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| prefix tokenadv totalnorm errtok | `Instruments-grec-grpo-prefix-tokenadv-totalnorm-errtok-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` |

### A.3 Hint Ablation 结果目录

| Variant | 结果目录 |
| --- | --- |
| plain rule only | `Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| dynamic hint rule only | `Instruments-grec-grpo-rule-only-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495` |
| fixed hint rule only | `Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495` |
