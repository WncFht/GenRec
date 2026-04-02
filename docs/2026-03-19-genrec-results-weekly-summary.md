# GenRec 本周主结果周报摘要（2026-03-19）

- 汇报范围：`GenRec` 当前 `Instruments-grec` 主线 RL 实验
- 详细版： [2026-03-18-genrec-main-results-weekly.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-03-18-genrec-main-results-weekly.md)
- 文档性质：这是对 2026-03-18 详细周报的人工摘要，方便快速回看，不是单独维护的 source of truth。

- 基线 SFT：`GenRec/results/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495`

- 统一设置：`Qwen2.5-3B-Instruct` RL, `qwen4B-4-256` semantic ID, `split=grec`

- 文中“相对提升(%)”统一按 `(当前值 - 对照值) / 对照值 * 100%` 计算。

## 1. 结论

- 继续做 `prefix reward` 实验，但是没有拿到收益。

- 做了一下 `ndcg reward` 的消融。

- 最好的 `NDCG@10` 结果来自 `rule_only rerun`，最佳点为 `checkpoint-2997`，`NDCG@10=0.0960`。

- 如果只看 top-10，`rule_only` 比当前 baseline mixed GRPO 还略强；但它和 baseline 一样，都存在 `HR@50` 低于 SFT 的问题。

- 实现了 `hint scaffold` 的功能，跑了一个离线和一个在线的实验。

- 仍然是 `fixed hint + rule_only`：

  - 按 `NDCG@10` 选 best checkpoint，最佳点更新为 `checkpoint-3326`
  - `NDCG@10=0.0953`
  - `HR@10=0.1193`
  - `NDCG@50=0.1114`
  - `HR@50=0.1938`

- 如果单独看 coverage 峰值，`fixed hint` 在 `checkpoint-2331` 仍然达到当前最高 `HR@50=0.1957`。

- 在线实验还有点问题。
