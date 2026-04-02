# 2026-04-01 Games-grec 全流程实验记录（qwen3-embedding-4B index + Qwen2.5-3B SFT/RL）

- 记录日期：2026-04-01
- 维护日期：2026-04-02
- 记录目的：把 `Games` 数据集从 `index -> grec preprocess -> SFT -> 后续 RL / evaluate` 的关键信息集中到一份文档里，后面继续往同一篇里补 `rule_only`、`fixed_hint` 等 RL 结果。
- 当前阶段：`index` 已训练并导出，`Games_grec` 数据已构建，`Games-grec` SFT 已完成，RL 尚未开始。
- 本地核对：截至 `2026-04-02`，`results/` 下还没有 `*Games*` 结果目录，因此“RL 尚未开始”与当前本地状态一致。
- 相关 W&B：
  `https://wandb.ai/wncfht/MIMIGenRec-SFT/runs/2trpxzle`

## 一句话结论

`Games` 单数据集的 `qwen3-embedding-4B + rq4(cb256x4)` semantic index 已成功训练并导出，最终 generate 阶段碰撞率为 `0.0075873`；基于该 index 的 `Games_grec` 下游数据也已构建完成。随后 `Games-grec-sft-qwen4B-4-256-dsz0` 完成到 `checkpoint-1024`，本地保存的最终汇总结果显示 `eval_loss=1.5389`、`train_loss=1.0617`，训练日志末尾对应 `epoch=4.0157`、`current_steps=1024`。接下来可以直接进入 `rule_only` 和 `fixed_hint` 的 RL 训练与统一评测。

## 1. 本地整理后的材料

本次下载回本地并整理进素材目录的文件：

- `docs/assets/2026-04-01-games-grec-qwen4b-4-256/all_results.json`
- `docs/assets/2026-04-01-games-grec-qwen4b-4-256/trainer_log.jsonl`
- `docs/assets/2026-04-01-games-grec-qwen4b-4-256/sft_wandb_loss_curves.png`

后面如果再下载：

- `trainer_state.json`
- `train_results.json`
- `eval_results.json`
- RL 的 `metrics.json`

建议也统一放到这个目录里，不要再散落到 `docs/` 根目录。

## 2. Index 训练结果

### 2.1 配置

| 项目 | 值 |
| --- | --- |
| Dataset | `Games` |
| Embedding model | `qwen3-embedding-4B` |
| Index config | `rq4_cb256-256-256-256` |
| `sk_eps` | `0.0-0.0-0.0-0.003` |
| Batch size | `256` |
| LR | `0.001` |
| Epoch | `500` |
| KMeans tag | `kmtrue-lkmtrue-kmi100` |
| 训练目录 | `index_train_runs/Games/index/qwen3-embedding-4B/rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003_kmtrue-lkmtrue-kmi100/Apr-01-2026_19-21-03/` |

### 2.2 核心结果

| 指标 | 值 |
| --- | ---: |
| All indices number | `13839` |
| Unique indices | `13734` |
| Max number of conflicts | `5` |
| Final collision rate | `0.007587253414264037` |
| Training best collision rate | `0.04754678806272129` |
| Avg utilization | `0.7959` |
| Layer 0 utilization | `47 / 256` |
| Layer 1/2/3 utilization | `256 / 256` |

### 2.3 导出产物

- 完整带 run-id 的导出：
  `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data/Games/Games.index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames_ridApr-01-2026-19-21-03.json`
- 稳定 alias：
  `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data/Games/Games.index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames.json`

可直接引用的观察：

- generate 阶段最终只剩约 `105` 个 item 还发生冲突，说明这个 `Games` cb256 index 已经足够进入下游 SFT/RL。
- codebook 利用率明显不均衡，第 0 层严重欠利用，后 3 层全部打满；这后面值得作为单独分析点。

## 3. GREC 下游数据构建结果

### 3.1 构建入口

使用的脚本是：

```bash
bash scripts/run_games_preprocess_grec.sh build
```

对应数据变体：

- `Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames`

### 3.2 样本规模

#### 原始 split

| Split | Rows |
| --- | ---: |
| Train | `246737` |
| Valid | `42259` |
| Test | `42259` |

#### 最终导出规模

| 输出 | 样本数 |
| --- | ---: |
| SFT train | `520761` |
| SFT valid | `42259` |
| SFT test | `42259` |
| RL train | `280110` |
| RL valid | `42259` |
| RL test | `42259` |

#### 任务拆分

| Task | 说明 | 样本数 |
| --- | --- | ---: |
| Task1 `SidSFT` | `sft, rl, train, valid, test` | train `246737`, valid `42259`, test `42259` |
| Task2 `SidItemFeat` | `sft, train only` | `27287` |
| Task3 `FusionSeqRec` | `sft, history_sids -> title` | train `246737` |
| Task4 `Title2Sid` | `rl, train only` | `10000` |
| Task5 `TitleDesc2Sid` | `rl, train only` | `23373` |

### 3.3 下游关键路径

- `new_tokens.json`：
  `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames/new_tokens.json`
- `id2sid.json`：
  `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames/id2sid.json`
- `sft/train.json`：
  `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames/sft/train.json`
- `rl/train.json`：
  `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames/rl/train.json`

## 4. Games-grec SFT 结果

### 4.1 训练入口与输出目录

使用配置：

- YAML：
  `examples/train_full/Games/games_rec_full_sft_3b_dsz0_qwen4b_4_256_grec.yaml`
- 输出目录：
  `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Games-grec-sft-qwen4B-4-256-dsz0`
- W&B run：
  `https://wandb.ai/wncfht/MIMIGenRec-SFT/runs/2trpxzle`

从你给的远端目录看，当前已有这些 checkpoint：

- `checkpoint-128`
- `checkpoint-256`
- `checkpoint-384`
- `checkpoint-512`
- `checkpoint-640`
- `checkpoint-768`
- `checkpoint-896`
- `checkpoint-1024`

### 4.2 本地汇总结果

来自 `docs/assets/2026-04-01-games-grec-qwen4b-4-256/all_results.json`：

| 指标 | 值 |
| --- | ---: |
| `epoch` | `4.015728680265421` |
| `eval_loss` | `1.5388939380645752` |
| `train_loss` | `1.0617161795380525` |
| `train_runtime` | `30410.2605 s` |
| `train_samples_per_second` | `171.245` |
| `train_steps_per_second` | `0.084` |
| `eval_runtime` | `141.3615 s` |
| `eval_samples_per_second` | `298.943` |
| `eval_steps_per_second` | `2.342` |
| `total_flos` | `1.2549072324726358e+19` |

### 4.3 训练日志摘记

来自 `docs/assets/2026-04-01-games-grec-qwen4b-4-256/trainer_log.jsonl`：

| 项目 | 值 |
| --- | ---: |
| 记录条数 | `1033` |
| 纯 train loss 记录数 | `1024` |
| eval loss 记录数 | `8` |
| 最小 train loss | `0.5094`（step `1023`） |
| 最后一个 train loss | `0.5165`（step `1024`） |
| 最后一个 eval loss | `1.5388939380645752`（step `1024`） |
| 最后记录的 epoch | `4.015728680265421` |
| 最后记录的 current step | `1024` |
| 总计划 step | `2550` |

可直接写进实验记录的观察：

- `train/loss` 从开头的 `9.6+` 很快下降到 `2` 以下，后段稳定在 `0.5x`。
- `eval/loss` 从图上看先快速下降，之后在 `1.3x ~ 1.5x` 区间震荡。
- 当前本地汇总文件只覆盖到 `checkpoint-1024` / `epoch≈4.02`，因此这次 SFT 结果记录应该按“训练在第 4 个 epoch 左右结束/停止”来表述，不要误写成跑满了配置里的 `10 epoch`。

### 4.4 W&B 曲线截图

下图来自你给的 W&B 页面截图，已经归档到：
`docs/assets/2026-04-01-games-grec-qwen4b-4-256/sft_wandb_loss_curves.png`

![Games-grec SFT train/eval loss curves](assets/2026-04-01-games-grec-qwen4b-4-256/sft_wandb_loss_curves.png)

从截图目测可以补一句：

- `eval/loss` 在前期大幅下降，2 到 3 epoch 左右达到较低区间，后段略有回升；
- `train/loss` 则持续下降并在后段进入缓慢收敛区。

### 4.5 已完成的 checkpoint 评测结果

截至 `2026-04-03`，本地 `results/Games-grec-sft-qwen4B-4-256-dsz0/` 下已经有 8 个 checkpoint 的评测结果：

| Checkpoint | NDCG@10 | HR@10 | NDCG@5 | HR@5 | NDCG@50 | HR@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-128` | `0.0036` | `0.0075` | `0.0026` | `0.0044` | `0.0059` | `0.0177` |
| `checkpoint-256` | `0.0184` | `0.0348` | `0.0144` | `0.0222` | `0.0320` | `0.0981` |
| `checkpoint-384` | `0.0326` | `0.0598` | `0.0258` | `0.0385` | `0.0536` | `0.1568` |
| `checkpoint-512` | `0.0377` | `0.0700` | `0.0292` | `0.0437` | `0.0622` | `0.1833` |
| `checkpoint-640` | `0.0426` | `0.0788` | `0.0333` | `0.0499` | `0.0680` | `0.1958` |
| `checkpoint-768` | `0.0433` | `0.0804` | `0.0336` | `0.0503` | `0.0691` | `0.1998` |
| `checkpoint-896` | `0.0430` | `0.0790` | `0.0340` | `0.0507` | `0.0687` | `0.1970` |
| `checkpoint-1024` | `0.0374` | `0.0692` | `0.0294` | `0.0443` | `0.0597` | `0.1729` |

可以直接记住这几个结论：

- 按 `NDCG@10` 看，当前最优点是 `checkpoint-768`，达到 `0.0433`。
- `checkpoint-896` 不是全局最优，但与 `768` 非常接近，且 `NDCG@5` / `HR@5` 略高。
- `checkpoint-1024` 出现了比较明显的回落，因此不建议再把它当作 RL 初始化点。

### 4.6 RL 起跑点选择

当前建议把 `checkpoint-896` 作为后续 RL 的初始化点。

原因：

- 它处在后段高性能区间，指标与 `768` 很接近，没有明显掉出平台期。
- 相比 `1024`，`896` 避开了末尾已经出现的退化。
- 如果目标是从一个“接近最优、但还没明显过拟合/回落”的点继续做 RL，`896` 是比 `1024` 更稳妥的选择。

换句话说：

- 如果你想追求最强 SFT 单点基线，文档里应记 `768` 是当前 best `NDCG@10`；
- 如果你想挑一个更适合继续接 RL 的后段 checkpoint，当前可以优先用 `896`。

## 5. 现在能不能直接用统一评测脚本

可以。

当前 [`scripts/evaluate_all_checkpoints.sh`](/Users/fanghaotian/Desktop/src/GenRec/scripts/evaluate_all_checkpoints.sh) 已经支持：

- `Games` 默认数据映射
- `Games-grec*` 的自动 variant 识别

所以你现在要只评测这条 SFT 线，可以直接跑：

```bash
INCLUDE_SFT=1 \
INCLUDE_RL=0 \
MODEL_FILTER="Games-grec-sft-qwen4B-4-256-dsz0" \
bash scripts/evaluate_all_checkpoints.sh
```

我建议第一次先 dry-run 看计划：

```bash
INCLUDE_SFT=1 \
INCLUDE_RL=0 \
MODEL_FILTER="Games-grec-sft-qwen4B-4-256-dsz0" \
DRY_RUN=1 \
bash scripts/evaluate_all_checkpoints.sh
```

正常情况下，它会自动把 `Games-grec-sft-qwen4B-4-256-dsz0` 映射到：

- `data/Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames/sft/test.json`
- `data/Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames/id2sid.json`

## 6. 后续命令

### 6.1 先把 SFT 评测补上

```bash
INCLUDE_SFT=1 \
INCLUDE_RL=0 \
MODEL_FILTER="Games-grec-sft-qwen4B-4-256-dsz0" \
bash scripts/evaluate_all_checkpoints.sh
```

### 6.2 再跑 `rule_only` RL（从 `checkpoint-896` 开始）

```bash
bash hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only.sh \
  --model-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Games-grec-sft-qwen4B-4-256-dsz0/checkpoint-896
```

### 6.3 然后跑 `fixed_hint` RL（同样从 `checkpoint-896` 开始）

```bash
bash hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-fixed-hint.sh \
  --model-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Games-grec-sft-qwen4B-4-256-dsz0/checkpoint-896
```

### 6.4 RL 完成后统一评测

```bash
MODEL_FILTER="Games-grec" \
bash scripts/evaluate_all_checkpoints.sh
```

## 7. 当前还缺什么

这篇文档现在已经覆盖了：

- `Games` index 训练
- `Games_grec` 数据构建
- `Games-grec` SFT 训练完成状态
- SFT 评测入口
- RL 下一步命令

后面还需要继续补：

1. SFT 真正的 `metrics.json`
2. `rule_only` RL 的训练轨迹和 checkpoint 评测
3. `fixed_hint` RL 的训练轨迹和 checkpoint 评测
4. 三条线（SFT / RL rule_only / RL fixed_hint）的横向对比结论
