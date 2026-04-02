# GenRec 2026-03-19 以来 `results/` 实验盘点（Epoch 对齐版）

- 记录日期：2026-04-02
- 统计范围：当前本地 `results/` 中可读到 `metrics.json` 的实验目录
- 对照文档：
  - [2026-03-18-genrec-main-results-weekly.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-03-18-genrec-main-results-weekly.md)
  - [2026-03-19-genrec-results-weekly-summary.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-03-19-genrec-results-weekly-summary.md)
  - [genrec_rl_study_2026-03-28.tex](/Users/fanghaotian/Desktop/src/GenRec/docs/deepresearch/genrec_rl_study_2026-03-28/genrec_rl_study_2026-03-28.tex)

## 1. 结论先说

从 2026-03-19 之后到现在，`results/` 里真正推进主线结论的实验，不是继续堆新的 reward form，而是三类东西：

1. `Instruments-grec` 上的 corrected hint 线，尤其是 `fixedhint-taskfix` 和 `fixedhint-taskfix-sid-only`。
2. `Instruments-grec` 的 semantic ID codebook sweep，尤其是 `qwen4B-4-512` 的 SFT 结果。
3. `Instruments-mimionerec` 和 `Industrial_and_Scientific` 这两条跨数据集验证线。

如果只看当前最值得在组会上强调的增量，我建议用下面三句：

- `rule_only rerun` 仍然是 `Instruments-grec` 上 top-10 最强的 RL 线，best 点在 `epoch≈1.802`，`NDCG@10=0.0960`，但 `HR@50=0.1681` 仍低于 SFT-256。
- corrected fixed-hint 线是 3 月 19 日之后最有价值的新结果，尤其是 `fixedhint-taskfix-b16-sid-only` 在 `epoch≈1.805` 达到 `0.0945 / 0.1201 / 0.1103 / 0.1933`，只比 `rule_only` 少 `0.0015` 的 `NDCG@10`，却多 `0.0252` 的 `HR@50`。
- `Instruments-grec-sft-qwen4B-4-512-dsz0` 的 best 点在 `epoch≈9.000` 达到 `0.0955 / 0.1198 / 0.1122 / 0.1972`，已经接近甚至局部强于 256 主线 RL，说明 semantic ID 容量本身已经是需要单独控制的重要因素。

## 2. 这份报告怎样定义“3 月 19 日以后”

只看文件修改时间并不可靠，因为本地 `results/` 很多目录是在 2026-04-01 一次性同步回来的，目录 mtime 会一起变化。这里采用的是两层口径：

1. 用 [2026-03-18-genrec-main-results-weekly.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-03-18-genrec-main-results-weekly.md) 和 [2026-03-19-genrec-results-weekly-summary.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-03-19-genrec-results-weekly-summary.md) 作为“截至 3 月 19 日已经讲过什么”的边界。
2. 再用 [genrec_rl_study_2026-03-28.tex](/Users/fanghaotian/Desktop/src/GenRec/docs/deepresearch/genrec_rl_study_2026-03-28/genrec_rl_study_2026-03-28.tex#L157) 里对 old / rerun / gather-fix / taskfix / sid-only 的分类，区分：
   - 3 月 19 日之前就已经形成的 anchor 结果
   - 3 月 19 日之后新增或修正的方法线
   - backup / legacy / 排错目录

因此，这份报告不是单纯按 mtime 排序，而是按“和 3 月 19 日周报相比新增了什么实验线、修正了什么结论”来写。

## 3. Epoch 对齐口径

这份报告统一用 epoch，不再用 checkpoint step。

但当前本地 `results/` 里没有训练态的 `trainer_state.json`，所以不能直接读原始 epoch。这里采用仓库内部已经在用的同一套归一化口径：[`eval_wandb_sidecar.py`](/Users/fanghaotian/Desktop/src/GenRec/eval_wandb_sidecar.py#L677) 用

```python
epoch_progress = ckpt.step / max_checkpoint_step * model.num_train_epochs
```

来把 checkpoint step 映射成 epoch 进度。

本报告完全沿用这个口径：

- RL run：`num_train_epochs=2`
  - 来源：`hope/` 下各 RL 启动脚本默认值，例如 [`Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh`](/Users/fanghaotian/Desktop/src/GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh#L108)
  - corrected dynamic / fixed / sid-only 脚本也是同样的 `2`
- SFT run：`num_train_epochs=10`
  - 来源：对应 YAML，例如 [`instruments_rec_full_sft_3b_dsz0_qwen4b_4_256_grec.yaml`](/Users/fanghaotian/Desktop/src/GenRec/examples/train_full/Instruments/instruments_rec_full_sft_3b_dsz0_qwen4b_4_256_grec.yaml)

因此，下面所有 `epoch≈x.xxx` 都应理解为：

- 与项目侧边上传和后续 W&B epoch 轴一致的 `epoch_progress`
- 不是训练器原始逐步记录的 `epoch`

这个近似在当前没有 `trainer_state.json` 的本地结果树里，是最一致、也最不容易混淆不同 run 的做法。

## 4. `Instruments-grec`：把 3 月 19 日之前的 anchor 先钉住

3 月 19 日周报已经讲清楚的 anchor 结论，到今天仍然成立：

| Anchor | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 | 读法 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Instruments-grec-sft-qwen4B-4-256-dsz0` | 7.857 | 0.0823 | 0.1094 | 0.0985 | 0.1844 | 256 主线 SFT 基线 |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.001 | 0.0952 | 0.1145 | 0.1071 | 0.1696 | mixed GRPO 很早就见顶，coverage 回落 |
| `Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.802 | 0.0960 | 0.1179 | 0.1070 | 0.1681 | top-10 最强，但 top-50 coverage 低 |
| `Instruments-grec-grpo-prefix-tokenadv-totalnorm-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` | 2.000 | 0.0906 | 0.1125 | 0.1054 | 0.1809 | reward shaping 里最好，但仍没超过 `rule_only` |
| `Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495` | 2.000 | 0.0953 | 0.1193 | 0.1114 | 0.1938 | 3 月 19 日前最平衡的 fixed hint 结果 |

对齐成 epoch 以后，有一个比 checkpoint 视角更清楚的现象：

- baseline mixed GRPO 的 best 点只在 `epoch≈1.0`
- `rule_only rerun` 的 best 点在 `epoch≈1.8`
- old fixed hint mixed-single 则一直涨到 `epoch≈2.0`

这意味着 3 月 19 日之前其实已经能看出一个趋势：真正能把后半程训练利用起来的，不是更 dense 的 reward，而是 hint scaffold。

## 5. `Instruments-grec`：3 月 19 日之后最重要的新实验，是 corrected hint 线

### 5.1 结果表

| Variant | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 | 读法 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dynamic hint rule-only (old)` | 1.001 | 0.0919 | 0.1160 | 0.1080 | 0.1905 | 3 月 19 日前的在线 dynamic anchor |
| `dynamic hint + gather fix` | 1.802 | 0.0936 | 0.1169 | 0.1083 | 0.1855 | 修完 gather 后 best 点延后到后半程 |
| `dynamic hint + ranking` | 1.001 | 0.0911 | 0.1138 | 0.1058 | 0.1822 | ranking 没带来净收益 |
| `dynamic hint sid-only` | 1.805 | 0.0921 | 0.1155 | 0.1065 | 0.1830 | 比 ranking 好，但仍弱于 gather-fix |
| `fixed hint task+index early` | 2.000 | 0.0894 | 0.1131 | 0.1051 | 0.1864 | task+index 修复后的早期短跑验证 |
| `fixed hint taskfix b16` | 1.802 | 0.0931 | 0.1189 | 0.1094 | 0.1941 | corrected fixed-hint 主干版 |
| `fixed hint taskfix b16 sid-only` | 1.805 | 0.0945 | 0.1201 | 0.1103 | 0.1933 | 当前最干净、最平衡的 corrected 结果 |

### 5.2 主要判断

#### 结论 1：corrected dynamic 确实修好了“训练后半段无效”问题，但还没赢 corrected fixed

`dynamic hint rule-only (old)` 的 best 点在 `epoch≈1.001`，而 `dynamic hint + gather fix` 的 best 点延后到了 `epoch≈1.802`。这说明 3 月 20 日之后的 gather 修复，至少把 dynamic 线从“中前期见顶”修到了“后半程还能继续涨”。

但 corrected dynamic 的最终高度仍然不够：

- `dynamic + gather fix`：`NDCG@10=0.0936`，`HR@50=0.1855`
- `fixed taskfix b16 sid-only`：`NDCG@10=0.0945`，`HR@50=0.1933`

也就是说，corrected dynamic 虽然更稳定了，但 corrected fixed 仍然更平衡。

#### 结论 2：ranking 方向没有给出正证据

`dynamic hint + ranking` 的 best 点停在 `epoch≈1.001`，四个主指标都弱于 `dynamic + gather fix`：

- `NDCG@10`: `0.0911 < 0.0936`
- `HR@10`: `0.1138 < 0.1169`
- `NDCG@50`: `0.1058 < 0.1083`
- `HR@50`: `0.1822 < 0.1855`

因此，至少在当前代码和当前训练口径下，`ranking` 更像一个没有证明收益的支线，而不是值得继续主打的方法。

#### 结论 3：`fixedhint-taskfix-b16-sid-only` 是 3 月 19 日之后最值得汇报的新主结果

`fixedhint-taskfix-b16-sid-only` 在 `epoch≈1.805` 达到：

- `NDCG@10=0.0945`
- `HR@10=0.1201`
- `NDCG@50=0.1103`
- `HR@50=0.1933`

和 `rule_only rerun` 相比：

- `NDCG@10` 只少 `0.0015`
- `HR@10` 反而多 `0.0022`
- `NDCG@50` 多 `0.0033`
- `HR@50` 多 `0.0252`

如果你只想选一条“3 月 19 日以后新增、而且现在最适合写进主线 narrative”的 run，我建议优先讲这条，而不是继续讲 old fixed mixed-single。原因不是 old mixed-single 不强，而是 `taskfix + sid-only` 这条线更贴近 corrected 方法定义，论述上更干净。

#### 结论 4：epoch 视角下，good hint run 的共同特征是 best 点都在后半程

把 corrected hint 家族统一换成 epoch 之后，一个很清楚的结构出现了：

- old dynamic：best 在 `epoch≈1.0`
- ranking dynamic：best 在 `epoch≈1.0`
- corrected dynamic：best 在 `epoch≈1.8`
- corrected fixed：best 在 `epoch≈1.8~2.0`

换句话说，3 月 19 日之后真正有效的修复，不是“把峰值抬高一点”，而是“把原来浪费掉的后半个 epoch 变成有用训练”。

## 6. `Instruments-grec`：3 月 19 日之后还有一条很重要的线是 SFT codebook sweep

### 6.1 结果表

| SFT Variant | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen4B-4-32` | 8.596 | 0.0727 | 0.0968 | 0.0884 | 0.1699 |
| `qwen4B-4-64` | 8.333 | 0.0630 | 0.0978 | 0.0788 | 0.1707 |
| `qwen4B-4-128` | 10.000 | 0.0628 | 0.0899 | 0.0805 | 0.1713 |
| `qwen4B-4-256` | 7.857 | 0.0823 | 0.1094 | 0.0985 | 0.1844 |
| `qwen4B-4-512` | 9.000 | 0.0955 | 0.1198 | 0.1122 | 0.1972 |

### 6.2 主要判断

#### 结论 1：`qwen4B-4-512` SFT 本身已经非常强

`Instruments-grec-sft-qwen4B-4-512-dsz0` 在 `epoch≈9.000` 达到：

- `NDCG@10=0.0955`
- `HR@10=0.1198`
- `NDCG@50=0.1122`
- `HR@50=0.1972`

它和 256 主线上的两个代表性 RL 结果对比：

- 相比 `rule_only rerun`：
  - `NDCG@10` 还高 `+0.0005`
  - `HR@50` 高 `+0.0291`
- 相比 old `fixed hint mixed-single`：
  - `NDCG@10` 只高 `+0.0002`
  - `HR@50` 高 `+0.0034`

这条结果不能直接拿来否定 RL，因为 codebook 容量已经变了；但它明确说明了一件事：从 3 月 19 日以后看结果，不应该只盯着 RL 设计本身，semantic ID 容量已经足以改写主比较面。

#### 结论 2：32/64/128 都没有接近 256，更别说 512

32/64/128 三条线的 best 点都在后期 epoch，但结果都明显弱于 256：

- `NDCG@10` 只在 `0.0628 ~ 0.0727`
- `HR@50` 只在 `0.1699 ~ 0.1713`

所以从当前本地结果看，`256 -> 512` 是明显有益的，而 `32/64/128` 都不能作为强 baseline。

## 7. Reward form 家族：3 月 19 日以后没有出现推翻主结论的新赢家

为了完整性，这里把 reward form 线也统一换成 epoch 再看一遍：

| Variant | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline mixed GRPO` | 1.001 | 0.0952 | 0.1145 | 0.1071 | 0.1696 |
| `rule_only rerun` | 1.802 | 0.0960 | 0.1179 | 0.1070 | 0.1681 |
| `prefix only` | 1.143 | 0.0862 | 0.1049 | 0.0982 | 0.1612 |
| `prefix seq only rerun` | 1.600 | 0.0857 | 0.1046 | 0.0985 | 0.1636 |
| `prefix token only` | 2.000 | 0.0815 | 0.1005 | 0.0958 | 0.1668 |
| `tokenadv raw` | 1.500 | 0.0476 | 0.0597 | 0.0564 | 0.0991 |
| `tokenadv totalnorm` | 2.000 | 0.0906 | 0.1125 | 0.1054 | 0.1809 |
| `tokenadv errtok` | 1.500 | 0.0796 | 0.0968 | 0.0914 | 0.1513 |

epoch 口径下，这几条线的关系和 3 月 19 日周报相比没有本质变化：

- `rule_only rerun` 仍然是 top-10 最强
- `tokenadv totalnorm` 仍然是唯一还算能站住的 shaping 线
- 其余 prefix 变体仍然无法替代 exact reward

如果一定要补一个新的 epoch 视角观察，就是：

- `baseline mixed GRPO` 在 `epoch≈1.0` 就见顶
- `rule_only rerun` 要到 `epoch≈1.8` 才见顶
- `tokenadv totalnorm` 则拖到了 `epoch≈2.0`

这说明 dense reward 并没有自动换来“更长的有效训练区间”，反而是 `rule_only` 和 corrected hint 更能把第二个 epoch 利用起来。

## 8. 跨数据集验证：`mimionerec` 和 `Industrial_and_Scientific` 复现了同一种 trade-off

### 8.1 结果表

| Dataset | Variant | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Industrial_and_Scientific` | SFT `dsz0-4gpu-eq8` | 8.125 | 0.0956 | 0.1363 | 0.1147 | 0.2241 |
| `Industrial_and_Scientific` | RL | 1.802 | 0.1035 | 0.1381 | 0.1211 | 0.2191 |
| `Instruments-mimionerec` | SFT `dsz0-2` | 8.182 | 0.1243 | 0.1617 | 0.1446 | 0.2554 |
| `Instruments-mimionerec` | RL | 1.000 | 0.1258 | 0.1587 | 0.1415 | 0.2311 |

### 8.2 主要判断

#### 结论 1：`Industrial_and_Scientific` 和 `mimionerec` 都在重复“top-10 升、coverage 降”的结构

`Industrial_and_Scientific`：

- RL 相比 SFT，`NDCG@10` 提升 `+0.0079`
- 但 `HR@50` 下降 `-0.0050`

`mimionerec`：

- RL 相比 SFT，`NDCG@10` 只提升 `+0.0015`
- 但 `HR@50` 下降 `-0.0243`

这说明 `Instruments-grec` 上看到的 coverage 收缩，不是单数据集偶然现象，而更像当前 RL 目标的普遍形态。

#### 结论 2：`mimionerec` 的 RL 峰值更早，说明它未必需要完整两轮

`Instruments-mimionerec-grpo...` 的 best 点出现在 `epoch≈1.000`，而不是接近 2。和 `Instruments-grec` 上 corrected hint 在 `1.8~2.0 epoch` 见顶形成对比。

这意味着：

- 不是所有数据集都适合“第二个 epoch 继续训”
- 后面如果做跨数据集统一叙事，最好把“最佳 epoch 是否偏前”也作为一个观察维度，而不是只看 best checkpoint 编号

## 9. `Games-grec`：脚本和文档已经准备好，但当前本地 `results/` 里还没有 RL 结果

这周 `hope/` 里已经把 `Games-grec` 的 `rule_only` 和 `fixed_hint` 启动脚本补齐了：

- [`Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only.sh`](/Users/fanghaotian/Desktop/src/GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only.sh#L100)
- [`Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-fixed-hint.sh`](/Users/fanghaotian/Desktop/src/GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-fixed-hint.sh#L105)

但截至 [2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-01-games-grec-qwen4b-4-256-full-pipeline.md#L5) 和当前本地 `results/` 状态，`*Games*` 目录还不存在，所以这条线现在只能汇报成：

- pipeline ready
- SFT 已完成
- RL 尚未有本地评测结果

## 10. 当前最推荐的汇报口径

如果这周只讲最重要的五点，我建议直接按下面五句组织：

1. `Instruments-grec` 上，top-10 最强结果仍然是 `rule_only rerun`，best 点在 `epoch≈1.802`，`NDCG@10=0.0960`。
2. 3 月 19 日之后最有价值的新结果不是新的 reward shaping，而是 corrected fixed-hint，尤其是 `fixedhint-taskfix-b16-sid-only`，在 `epoch≈1.805` 几乎保住 top-10，同时把 `HR@50` 拉回到 `0.1933`。
3. corrected dynamic 通过 gather-fix 把 best 点从 `epoch≈1.0` 推迟到了 `epoch≈1.8`，说明修复有效，但它仍然没有超过 corrected fixed-hint。
4. `Instruments-grec-sft-qwen4B-4-512-dsz0` 已经达到 `0.0955 / 0.1972`，说明 semantic ID 容量本身已经足以改变主结果比较面。
5. `mimionerec` 和 `Industrial_and_Scientific` 也在重复“top-10 改善、HR@50 收缩”的结构，所以 coverage 问题不是 `Instruments-grec` 的偶发现象。

## 附录 A：完整实验清单（按当前本地 `results/` 可见指标）

### A.1 `Instruments-grec` SFT

| Experiment | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Instruments-grec-sft-qwen4B-4-32-dsz0` | 8.596 | 0.0727 | 0.0968 | 0.0884 | 0.1699 |
| `Instruments-grec-sft-qwen4B-4-64-dsz0` | 8.333 | 0.0630 | 0.0978 | 0.0788 | 0.1707 |
| `Instruments-grec-sft-qwen4B-4-128-dsz0` | 10.000 | 0.0628 | 0.0899 | 0.0805 | 0.1713 |
| `Instruments-grec-sft-qwen4B-4-256-dsz0` | 7.857 | 0.0823 | 0.1094 | 0.0985 | 0.1844 |
| `Instruments-grec-sft-qwen4B-4-512-dsz0` | 9.000 | 0.0955 | 0.1198 | 0.1122 | 0.1972 |

### A.2 `Instruments-grec` Reward / Baseline RL

| Experiment | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.001 | 0.0952 | 0.1145 | 0.1071 | 0.1696 |
| `Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.802 | 0.0960 | 0.1179 | 0.1070 | 0.1681 |
| `Instruments-grec-grpo-prefixonly-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.143 | 0.0862 | 0.1049 | 0.0982 | 0.1612 |
| `Instruments-grec-grpo-prefix-seq-only-fixbool-rerun-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.600 | 0.0857 | 0.1046 | 0.0985 | 0.1636 |
| `Instruments-grec-grpo-prefix-token-only-totalnorm-qwen2.5-3b-qwen4B-4-256-from-sft495` | 2.000 | 0.0815 | 0.1005 | 0.0958 | 0.1668 |
| `Instruments-grec-grpo-prefix-tokenadv-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.500 | 0.0476 | 0.0597 | 0.0564 | 0.0991 |
| `Instruments-grec-grpo-prefix-tokenadv-totalnorm-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` | 2.000 | 0.0906 | 0.1125 | 0.1054 | 0.1809 |
| `Instruments-grec-grpo-prefix-tokenadv-totalnorm-errtok-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.500 | 0.0796 | 0.0968 | 0.0914 | 0.1513 |

### A.3 `Instruments-grec` Hint / Corrected Hint / Legacy

| Experiment | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Instruments-grec-grpo-rule-only-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.001 | 0.0919 | 0.1160 | 0.1080 | 0.1905 |
| `Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.802 | 0.0936 | 0.1169 | 0.1083 | 0.1855 |
| `Instruments-grec-grpo-ranking-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.001 | 0.0911 | 0.1138 | 0.1058 | 0.1822 |
| `Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.805 | 0.0921 | 0.1155 | 0.1065 | 0.1830 |
| `Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495` | 2.000 | 0.0953 | 0.1193 | 0.1114 | 0.1938 |
| `Instruments-grec-grpo-rule-only-fixed-hint-task-index-fix-beam16-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495` | 2.000 | 0.0894 | 0.1131 | 0.1051 | 0.1864 |
| `Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495` | 1.802 | 0.0931 | 0.1189 | 0.1094 | 0.1941 |
| `Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495` | 1.805 | 0.0945 | 0.1201 | 0.1103 | 0.1933 |
| `Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495-backup` | 2.000 | 0.0877 | 0.1121 | 0.1030 | 0.1834 |
| `Instruments-grec-grpo-rule-only-qwen2.5-3b-qwen4B-4-256-from-sft495` | 0.667 | 0.0761 | 0.0907 | 0.0885 | 0.1494 |
| `Instruments-grec-grpo-prefix-qwen2.5-3b-qwen4B-4-256-from-sft495` | 1.667 | 0.0863 | 0.1060 | 0.0985 | 0.1631 |
| `Instruments-grec-grpo-prefix-qwen2.5-3b-qwen4B-4-256-from-sft495-backup` | 2.000 | 0.0815 | 0.0998 | 0.0946 | 0.1608 |
| `Instruments-grec-grpo-prefix-seq-only-qwen2.5-3b-qwen4B-4-256-from-sft495` | 0.801 | 0.0389 | 0.0494 | 0.0504 | 0.1023 |

### A.4 Cross-Dataset

| Experiment | Best Epoch | NDCG@10 | HR@10 | NDCG@50 | HR@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Industrial_and_Scientific-sft-dsz0-4gpu-eq8` | 8.125 | 0.0956 | 0.1363 | 0.1147 | 0.2241 |
| `Industrial_and_Scientific-qwen2.5-3b-instruct-grpo` | 1.802 | 0.1035 | 0.1381 | 0.1211 | 0.2191 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0` | 9.091 | 0.1220 | 0.1626 | 0.1422 | 0.2548 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0-2` | 8.182 | 0.1243 | 0.1617 | 0.1446 | 0.2554 |
| `Instruments-mimionerec-sft-qwen4B-4-256-dsz0-backup` | 10.000 | 0.1196 | 0.1520 | 0.1371 | 0.2331 |
| `Instruments-mimionerec-grpo-qwen2.5-3b-qwen4B-4-256-from-sft500` | 1.000 | 0.1258 | 0.1587 | 0.1415 | 0.2311 |

### A.5 `Games-grec`

当前本地 `results/` 里没有 `*Games*` 目录，因此没有可纳入本表的评测结果。
