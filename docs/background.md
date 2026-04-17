# GenRec 项目背景

- 最后更新：2026-04-18
- 维护目的：沉淀相对稳定、可复用的项目背景信息；一次性实验进展放到 [progress.md](progress.md)，带日期的阶段性总结继续放在 `docs/YYYY-MM-DD-*.md`。

## 1. 这个项目在做什么

`GenRec` 当前主要在做基于 semantic ID 的生成式推荐。核心做法是：

1. 先用 embedding 模型和量化 codebook 为 item 构造多层 semantic ID（SID）。
2. 再把推荐任务改写成“根据用户历史或 item 文本，生成目标 item 的 SID”。
3. 训练上分成 `SFT -> RL -> 统一评测` 三段，并持续对比不同 reward 形式和 hint/scaffold 设计。

当前文档的主研究重心是：

- `Instruments-grec`
- `Games-grec`

`Arts` 在仓库里已经有基础脚本支持，但截至本次整理还没有看到成体系的实验记录或结果目录。

## 2. 当前常用任务与数据口径

从 [sft_rl_data_pipeline.md](sft_rl_data_pipeline.md) 和多篇实验笔记来看，当前训练/评测围绕这些任务：

- `task1_sid_sft`
  含义：历史 SID 序列 -> 下一个 SID。
- `task4_hisTitle2sid`
  含义：历史标题序列 -> 下一个 SID。
- `task5_title_desc2sid`
  含义：标题/描述 -> SID。

SFT 侧还会包含：

- `task2`：SID 和标题之间的问答映射。
- `task3`：历史 SID -> 下一个标题。

实验里反复出现的一条经验是：这几个任务的难度并不一样，尤其 `hisTitle2sid` 通常比纯 `sid` 更难，所以很多结论必须做 task-level 拆分，不能只看 aggregate 指标。

## 3. 标准实验流水线

当前仓库里相对稳定的实验流水线可以概括成：

1. `index`
   用 embedding + quantization 训练 semantic index，导出 `*.index.json`。
2. `preprocess`
   把原始数据转成 `grec` 变体，生成 `new_tokens.json`、`id2sid.json`、`sft/*.json`、`rl/*.json`。
3. `SFT`
   先在生成任务上做监督微调，并选一个适合继续接 RL 的 checkpoint。
4. `RL`
   对比 `rule_only`、prefix 系列 reward、dynamic hint、fixed hint、CE 辅助项等变体。
5. `evaluate`
   用统一 checkpoint 评测脚本把不同 run 拉到同一指标口径上比较。
6. `upload/report`
   通过 [eval_wandb_sidecar.md](eval_wandb_sidecar.md) 维护远端 results 到本地 W&B 上传的流程。

补充约束：

- 本机主要用于代码整理、CPU 分析和写文档。
- 真正的训练和大规模生成主要在远端路径 `/mnt/dolphinfs/.../GenRec` 下进行。
- 因此文档里出现的远端路径本身就是实验背景的一部分，不应省略。

## 4. 当前研究主线

综合 [2026-03-18-genrec-main-results-weekly.md](2026-03-18-genrec-main-results-weekly.md)、[2026-04-11-genrec-instruments-rl-variant-comparison.md](2026-04-11-genrec-instruments-rl-variant-comparison.md)、[2026-04-16-instruments-dynamic-hint-max1-ablation.md](2026-04-16-instruments-dynamic-hint-max1-ablation.md) 和 `deepresearch` 综合稿，可以把当前项目主线概括成四条。

### 4.1 `rule_only` 仍然是重要 baseline

- 在 `Instruments-grec` 上，plain `rule_only` 仍然是最强的无 hint top-10 baseline。
- 它说明“exact reward 太稀疏所以一定不行”这个直觉并不成立。
- 但它的典型代价是 coverage 收缩，尤其 `HR@50` 容易比 SFT 更低。

### 4.2 纯 reward shaping 不是当前最可信的突破点

- prefix 类 reward 做过一整轮对比，但结果普遍弱于 `rule_only`。
- token-level prefix reward 对 normalization 非常敏感。
- 目前更合理的结论不是“reward shaping 毫无价值”，而是“它不是当前最稳定、最值得主打的贡献点”。

### 4.3 hint scaffold 是当前最有希望的方向

- `fixed hint + rule_only` 基本保住了 top-10，同时更稳定地修复了 coverage 损失。
- `dynamic hint` 也有价值，但当前长期表现仍普遍落后于 clean fixed-hint baseline。
- 因此当前最值得继续解释和推进的问题，不是“再发明一个 reward”，而是“怎样给 hint，才能既缓解稀疏奖励，又不把训练分布推离最终无 hint 评测太远”。

### 4.4 离线 hint 分析已经给出强机制线索

基于 [2026-03-16-genrec-hint-local-bundle-findings.md](2026-03-16-genrec-hint-local-bundle-findings.md)、[2026-03-17-genrec-rollout-node-analysis.md](2026-03-17-genrec-rollout-node-analysis.md) 和 `deepresearch` 综合稿，当前已有几条比较稳定的机制判断：

- 难点主要集中在前两层 SID token，而不是整棵树“太深”。
- `hisTitle2sid` 的难度不能只用局部分支结构解释掉，文本输入本身也在引入额外歧义。
- 很多极难样本最终会集中到少数 branch/pathology，而不是均匀散开。
- 这解释了为什么 scaffold/hint 能明显改变训练信号密度。

## 5. 当前几条方法线怎么理解

### 5.1 fixed hint

- 训练前先用离线分析导出 per-sample hint depth map。
- 训练时直接把 oracle prefix hint 拼到 prompt 里。
- 这是当前最稳定、最容易解释的 scaffold 线。

### 5.2 dynamic hint

- 训练时在线 cascade：先不加 hint，没命中再逐层加 hint。
- 它更接近“按当前策略状态自适应给 scaffold”的想法。
- 但现阶段的长期表现和稳定性仍弱于 fixed hint。

### 5.3 CE / hint-CE

- 当前是附着在 fixed-hint 主线上的辅助项。
- 现有证据更支持“它能把训练轨迹拉平”这一作用，而不是直接成为 headline 提升来源。
- `hintce-2` 是当前最值得继续追的 CE 变体。

### 5.4 UFT-style curriculum

- 这是下一阶段最明确的待落地方向之一。
- 仓库里已有想法文档，但还不是完成态结果线。
- 其核心不是照搬整套 UFT，而是借它的 `hint curriculum + hinted-token SFT/CE loss` 思路，把 fixed-hint 的收益逐步迁移到更接近 no-hint 的训练分布上。

## 6. 如何阅读这个 `docs/` 目录

建议按下面的优先级读：

1. 先看稳定文档：
   [background.md](background.md),
   [progress.md](progress.md),
   [experiment_tracking.md](experiment_tracking.md),
   [runtime_paths.md](runtime_paths.md),
   [sft_rl_data_pipeline.md](sft_rl_data_pipeline.md),
   [eval_wandb_sidecar.md](eval_wandb_sidecar.md)
2. 再看带日期的阶段性实验记录：
   顶层 `2026-*-*.md`
3. 需要整体复盘时，再看：
   `docs/deepresearch/genrec_rl_study_2026-03-28/`

阅读约定：

- 顶层按日期命名的文档是“时间快照”，不一定会被回写成最新状态。
- 当旧正文与更新后的本地导出表不一致时，优先相信较新的 summary CSV 和最近维护过的综合文档。
- 同一实验线应尽量继续维护原文，而不是不断新建近重复文档。

## 7. 后续维护边界

适合继续写进这份文档的内容：

- 项目目标、方法总览、长期有效的任务口径
- 稳定下来的方法判断
- 文档组织方式

不适合写进这份文档的内容：

- 某次单独 run 的最新 checkpoint 结果
- 还没收敛的对比数字
- 只对当前一轮实验排期有效的 TODO

这些内容应写进 [progress.md](progress.md) 或对应的 dated note。
