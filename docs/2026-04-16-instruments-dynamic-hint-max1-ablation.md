# Instruments Dynamic-Hint Max1 消融记录（2026-04-16）

- Record date: 2026-04-16
- Status: 已准备 launcher，等待明天跑完训练并同步结果与日志后做分析
- Goal: 在 `Instruments-grec` 上做一个最小改动的 `dynamic hint` 消融，只把 online hint budget 从 `<=3` 收紧到 `<=1`，不改 trainer 逻辑，不引入 no-hit 样本丢弃

## 1. 这次实验到底改了什么

这次实验只改一件事：

- 把 `dynamic_hint_max_depth` 从 `3` 改成 `1`

明确不改的部分：

- 不改 `reward_mode`，仍然是 `rule_only`
- 不改 `num_beams`，仍然是 `16`
- 不改 dataset variant，仍然是 `Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47`
- 不改 trainer 的 stage 收口逻辑
- 不做“给了 1 个 hint 还没 hit 就丢数据”的筛除

换句话说，这次实验验证的是：

> 如果只允许 `dynamic hint` 最多暴露 1 个 oracle token，而不是最多 3 个，最终的 train dynamics 和 eval trade-off 会怎么变。

它**不**验证：

> “给了 1 个 hint 后还没 hit 的样本是否应当被 drop”

后者如果要做，必须改 trainer，而这次没有做。

## 2. 对应脚本

- 新 launcher:
  [Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-max1.sh](/Users/fanghaotian/Desktop/src/GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-max1.sh)
- 参考原始 dynamic-hint gather-fix launcher:
  [Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh](/Users/fanghaotian/Desktop/src/GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh)

新脚本的默认关键配置是：

- `RUN_NAME=instruments_grec_rl_rule_only_dynamic_hint_max1_qwen2_5_3b_qwen4b_4_256_from_ckpt495`
- `OUTPUT_DIR=rl_outputs/Instruments-grec-grpo-rule-only-dynamic-hint-max1-qwen2.5-3b-qwen4B-4-256-from-sft495`
- `MAIN_PORT=29520`
- `DYNAMIC_HINT_MAX_DEPTH=1`
- `DYNAMIC_HINT_APPLY_TO_EVAL=false`

## 3. 建议运行方式

如果直接用默认配置，可以从仓库根目录启动：

```bash
bash hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-max1.sh --nohup
```

如果只想先确认命令拼接是否正确：

```bash
bash hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-max1.sh --dry-run
```

如果明天远端结果已经跑好，只需要把对应的 result dir 和 launcher log 拉回来即可。

## 4. 明天分析时的主对照线

这次 `max1` 消融最应该和下面几条线一起看。

### 4.1 主对照

- `dynamic hint gather-fix`
  - 当前 canonical dynamic baseline
  - best checkpoint: `checkpoint-2997`
  - best `NDCG@10=0.0936`
  - best-point `HR@50=0.1855`
- `dynamic hint sid-only`
  - 更偏 shallow conditioning 的现有 dynamic 参考
  - best checkpoint: `checkpoint-2394`
  - best `NDCG@10=0.0921`
  - best-point `HR@50=0.1830`
- `rule_only rerun`
  - no-hint exact reward baseline
  - best checkpoint: `checkpoint-2997`
  - best `NDCG@10=0.0960`
  - best-point `HR@50=0.1681`
- corrected `fixed hint taskfix sid-only`
  - 当前更强、也更稳定的 clean hint scaffold 参考线
  - best checkpoint: `checkpoint-2652`
  - best `NDCG@10=0.0945`
  - best-point `HR@50=0.1935`

上面的数字来自：

- [2026-04-11-genrec-instruments-rl-variant-comparison.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-11-genrec-instruments-rl-variant-comparison.md)

### 4.2 这次最关心的不是只有最终指标

`max1` 的直觉收益主要不是“直接把最终 best checkpoint 打穿”，而是看它是否能在下面几件事上更像一个合理的 hint budget：

1. `dynamic_hint/selected_hint_depth_mean` 是否明显下降
2. `dynamic_hint/selected_depth_1_frac` 是否成为主要停留层，而不是大量样本被推到更深 hint
3. `completions/mean_length` 是否更接近 no-hint 分布
4. `HR@50` 是否能比 plain `rule_only` 明显更高，同时不要比现有 dynamic baseline 掉得太狠
5. `NDCG@10` 是否仍然保住在可接受区间

## 5. 明天需要同步回来的东西

最少需要这几类材料：

1. launcher log
   - 至少要包含 train-time 标量打印与启动配置
   - 最重要的是能看到 `dynamic_hint_max_depth=1`
2. result dir
   - `checkpoint-*/metrics.json`
   - `trainer_state.json` 或同类训练态文件，如果有的话
3. 如果远端有更完整的 train log / wandb 导出，也一并带回来

如果只能拉最小集合，那最低要求是：

- 整个 run 的日志文件
- 对应 model result dir 下所有 `checkpoint-*/metrics.json`

## 6. 明天分析时我会优先看的指标

先看 train-time，再看 eval。

### 6.1 Train-time

- `dynamic_hint/selected_hint_depth_mean`
- `dynamic_hint/selected_depth_0_frac`
- `dynamic_hint/selected_depth_1_frac`
- `dynamic_hint/max_depth_miss_frac`
- `rewards/rule_reward/mean`
- `reward_std`
- `frac_reward_zero_std`
- `completions/mean_length`
- `kl`
- `entropy`

这里的解释重点是：

- 如果 `selected_hint_depth_mean` 明显下降，但 `max_depth_miss_frac` 爆炸，说明 hint budget 收得太狠
- 如果 `selected_hint_depth_mean` 下降，同时 `mean_length` 上升，通常说明模型需要自己补更长 suffix，这在机制上是合理的
- 如果 `reward_std` 或有效 reward 信号掉得太厉害，说明 `max1` 可能把太多样本压回了“难样本”区域

### 6.2 Eval-time

- `NDCG@10`
- `HR@10`
- `NDCG@50`
- `HR@50`
- best checkpoint 对比
- early-stop checkpoint 对比

这次 eval 最关键的问题不是“有没有刷新全场最好”，而是：

1. 它相对 `dynamic gather-fix` 的损失有多大
2. 它是否比 `rule_only rerun` 更平衡
3. 它是否在 top-10 / coverage trade-off 上更接近一个“浅 hint budget”而不是退化成 plain `rule_only`

## 7. 我对这次实验的预期

这是一个带方向性的预期，不是结论。

### 7.1 可能的正向结果

- `selected_hint_depth_mean` 会显著低于当前 `dynamic gather-fix`
- `completions/mean_length` 会更接近 no-hint 线
- `NDCG@10` 不一定最强，但可能比当前 dynamic 更接近 `rule_only`
- 如果 lucky，`HR@50` 仍能保住在 `dynamic sid-only` 附近

### 7.2 更可能出现的风险

- `max_depth_miss_frac` 上升
- `HR@50` 比 `dynamic gather-fix` 更低
- 一部分原来依赖 deeper hint 才能解的样本被迫回到 base / hint1 失败区

所以明天最值得防的误读是：

> 只看最终 `NDCG@10`，忽略它到底是不是通过更浅 hint budget 换来的更接近 no-hint 分布。

## 8. 明天我拿到结果后会怎么做

等你把 result 和 log 拉下来之后，我会按这个顺序分析：

1. 先确认这条 run 的配置没有漂移
   - 真的是 `dynamic_hint_max_depth=1`
   - 其它关键超参与 baseline 一致
2. 跑一版和现有 dynamic / fixed / rule-only 线并排的 checkpoint 表
3. 单独画 `selected_hint_depth_mean`、`selected_depth_0/1_frac`、`mean_length` 的对照图
4. 再判断它属于下面哪一类：
   - 成功的 shallow-budget dynamic
   - top-10 变好但 coverage 掉太多
   - coverage 仍可接受，但整体被现有 dynamic baseline 支配
   - 直接退化成近似 `rule_only`

## 9. 相关背景文档

- [Instruments RL 后续探索维护文档（2026-04-11）](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-11-instruments-rl-next-ideas.md)
- [GenRec Instruments RL 七线主比较 + fixed-hint CE 补充（2026-04-11）](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-11-genrec-instruments-rl-variant-comparison.md)
- [GenRec Dynamic Hint / Fixed Hint 指标与实现说明](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-03-17-genrec-dynamic-fixed-hint-metrics.md)
