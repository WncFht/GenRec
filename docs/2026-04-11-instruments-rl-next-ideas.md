# Instruments RL 后续探索维护文档（2026-04-11）

这份文档用于持续维护 `Instruments-grec` 上后续 RL 探索的主问题、假设和实验顺序。它不是一次性的结果汇报，而是后面继续推进时的工作底稿。

相关背景文档：

- [GenRec 本周主结果周报（Reward Form + Hint Ablation，2026-03-18）](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-03-18-genrec-main-results-weekly.md)
- [GenRec 周报（2026-04-02，Epoch 口径）](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-04-02-genrec-results-since-2026-03-19-epoch-report.md)
- [Dynamic / Fixed Hint 指标说明](/Users/fanghaotian/Desktop/src/GenRec/docs/2026-03-17-genrec-dynamic-fixed-hint-metrics.md)
- [genrec_rl_study_2026-03-28.tex](/Users/fanghaotian/Desktop/src/GenRec/docs/deepresearch/genrec_rl_study_2026-03-28/genrec_rl_study_2026-03-28.tex)

## 1. 当前可以当作稳定前提的结论

后续探索先默认以下判断成立，除非有新的结果直接推翻。

1. 无 hint 的 reward form 里，当前最强的 top-10 baseline 仍然是 `rule_only rerun`，不是更复杂的 prefix shaping。
2. `fixed hint` 仍然是当前最值得继续追的 scaffold 方向，因为它几乎保住 top-10，同时明显修复了 `rule_only` 的 coverage 损失。
3. old `fixed hint mixed-single` 不能直接当成“更正确的方法”，因为它吃到了 legacy `index-only` bug；但它依然提供了一个非常重要的现象学上界。
4. corrected `fixedhint-taskfix-b16` / `fixedhint-taskfix-b16-sid-only` 是更可信的工程基线。
5. `dynamic hint` 是有价值的，但当前 canonical 版本应以 `gather-fix` 后的 run 为准，而且它仍然落后于 fixed。
6. 继续盲目扩 reward form 的优先级不高。当前更值得打的是 hint curriculum、hint budget、以及 dynamic gate 本身。

一句话概括当前局面：

> 现在最值得解释和推进的，不是“再发明一个 reward”，而是“怎样给 hint 才既能缓解稀疏 reward，又不把训练分布推得离最终 no-hint eval 太远”。

## 2. 探索主线 A：把 UFT 引进来做 hint curriculum

参考项目：

- UFT 官方仓库：<https://github.com/liumy2010/UFT>

从 UFT 当前公开实现看，它最值得借的不是完整框架，而是两件事：

1. 它把 `hint` 直接拼到 prompt 里，而不是把 hint 当成额外 reward。
2. 它不是永远保持固定 hint，而是在训练前段保留 hint、后段逐步减少 hint，并在 actor loss 里额外加一个只作用在 hint 区域上的 SFT 项。

从源码看，UFT 的关键做法包括：

- 在 `RLHFDataset` 里按训练步数切 hint 长度，支持 uniform / stage / cosine-like 的 exposure 方式。
- 训练时记录 `hint_mask`，并对对应 prompt token 加一个额外的 log-likelihood loss。
- 显式暴露了 `T_hint` 和 `sft_loss_coef` 这样的控制量。

这和我们现在的问题非常对口，因为我们已经知道：

- fixed hint 的收益很大一部分来自 scaffold；
- 但 fixed hint 又存在一个核心 tension：
  - hint 给太深，训练更容易；
  - 但训练分布会离最终 no-hint eval 更远。

UFT-style curriculum 对我们最有吸引力的地方在于：

1. 它允许我们保留 fixed hint 的稳定性。
2. 它允许我们把 hint 当成“训练前期脚手架”，而不是全程不变的条件。
3. 它天然适合回答“hint 能不能逐步退场，但 coverage 不要立刻掉回去”这个问题。

### 2.1 最适合先试的 GenRec 版本

最合理的第一个落点不是 dynamic，而是：

- `fixedhint-taskfix-b16-sid-only`

原因：

1. 这是当前最干净、最可信的 fixed 基线之一。
2. `sid-only` 已经显示出较强的 top-10 hit。
3. 它能把“hint curriculum 本身”的效果，和“cross-task bug / dynamic gate”分开。

### 2.2 对 GenRec 的具体改法建议

建议把第一版范围压到最小：

1. hint 来源仍然使用当前的 task-aware fixed map。
2. 训练前半段保留当前 per-sample hint depth。
3. 训练后半段逐步把可见 hint 截短到更浅，最后逼近 no-hint。
4. 额外只对 hinted prefix 加一个轻量的 SFT 项，不动现有 `rule_only` reward 主体。

更具体一点，可以先试两条线：

- `fixed taskfix + hint anneal`
- `fixed taskfix + hint anneal + hint-token SFT loss`

先不要同时改太多别的东西。

### 2.3 我们真正想验证的不是“UFT 论文赢不赢”，而是这两个问题

1. fixed hint 的 coverage 收益，能不能在 hint 逐步退场后保住一部分。
2. old fixed 那种“更浅但更稳”的优势，能不能用一个显式 curriculum 复制出来，而不是继续依赖 bug。

### 2.4 第一批实验的成功标准

我建议先把成功标准写死，避免实验做完只剩主观感觉。

第一批 UFT-style 实验如果要算成功，至少满足下面之一：

1. 相比 `fixedhint-taskfix-b16-sid-only`，`NDCG@10` 不下降太多，但 `HR@50` 继续保住。
2. 相比固定深度版本，train-time `completions/mean_length` 更接近 no-hint 分布，同时最终 no-hint eval 不变差。
3. 相比 old fixed，它能复现“较浅 scaffold 带来的 top-10 好处”，但不再依赖 `index-only` bug。

## 3. 探索主线 B：先把 `completion_length` 这张图解释透

你给的 `train/completions/mean_length` 图，其实已经暴露出一个很关键的问题：

- old `fixed hint mixed-single` 基本构成了 hinted family 的上界，约在 `4.2` 左右；
- corrected `fixedhint-taskfix-b16` 基本构成了下界，约在 `3.95~4.0`；
- dynamic / ranking 线在这两者之间，并且会随训练推进往上爬；
- plain `rule_only` 则几乎钉在 `5`。

这里最值得先确认的一件事是：

> 这很可能首先是 hint depth 分布的投影，而不是模型“突然更喜欢生成长 completion”。

### 3.1 现在已经有一个很强的机制解释

`fixed_hint_grpo_trainer.py` 里记录的 `completions/mean_length`，统计的是最终选中的 `completion_ids_list` 长度，也就是模型实际生成的 suffix 长度，而不是“完整 semantic ID 长度 + hint”。

见：

- [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L270)

同时，从 old fixed bug 的 hint-depth 分布我们已经知道：

- correct `task+index` overall 平均 hint depth 约为 `1.036`
- legacy `index-only` overall 平均 hint depth 约为 `0.792`

数据来源：

- [fixed_hint_bug_depth_distribution.csv](/Users/fanghaotian/Desktop/src/GenRec/docs/deepresearch/genrec_rl_study_2026-03-28/data/fixed_hint_bug_depth_distribution.csv)

如果把 `plain rule_only` 那条几乎固定在 `5` 的线理解为“无 hint 条件下生成 4 个 semantic-ID token 加一个终止 token”，那么 hinted 版本的期望 suffix 长度大致应该满足下面这个近似：

```text
expected_completion_length ≈ 5 - avg_hint_depth
```

代进去会得到：

- correct task+index fixed: `5 - 1.036 = 3.964`
- legacy index-only fixed: `5 - 0.792 = 4.208`

这和图里的 lower / upper envelope 几乎直接对上了。

因此，目前最合理的第一判断不是：

- “修复版 fixed 学坏了，老 fixed 学好了”

而是：

- “修复版 fixed 平均给了更深的 hint，所以模型平均要补的 suffix 更短”
- “错版 fixed 平均给了更浅的 hint，所以模型平均要补的 suffix 更长”

也就是说，`completion_length` 这张图里，至少有很大一部分是 prompt construction 的直接结果。

### 3.2 这件事为什么重要

因为它会直接改变我们怎么读这类图。

如果不先除掉 hint depth 这个机械因素，就很容易把：

- 更深 hint 导致更短 suffix

误读成：

- 模型学习到了“更保守的长度偏好”

这是两回事。

### 3.3 这条线后面该怎么验证

优先做下面几件事：

1. 给 fixed 系列 run 也显式记录 train-time `oracle_hint_depth_mean`。
2. 把 `mean_length + mean_hint_depth` 一起画出来，检查是否近似守恒。
3. 对 dynamic run 对齐 `dynamic_hint/selected_hint_depth_mean` 和 `completions/mean_length`。
4. 按 task 拆开看，尤其是 `task4_hisTitle2sid`，因为 old fixed bug 对它压浅最狠。

如果这几项都成立，那么我们就能把 `completion_length` 这张图里的大部分现象，重新解释成：

- old fixed 比 corrected fixed 更浅；
- dynamic 后期往上爬，意味着它越来越常在更浅 stage 就停下来。

这反而更接近“是否在减少 hint 依赖”的问题，而不是“模型输出是不是坏了”的问题。

## 4. 探索主线 C：hint 上限要不要加，但不要做成太启发式

这里我同意你的直觉：

- 很可能应该限制 hint；
- 但不应该简单写成“最多 hint 1”这种硬编码启发式。

因为“最多 hint 1”虽然简单，但它把两个问题糊在一起了：

1. 我们是想限制 hint exposure。
2. 还是想改变 hardest samples 的训练难度。

这两个目标不一定一致。

### 4.1 比硬 cap 更合理的方向

我更倾向于把它写成“budget”而不是“rule”。

几个更合理的形式：

1. `hint depth penalty`
   - 训练目标里显式给 deeper hint 一个代价，例如对 depth 或 hint-token 数加惩罚。
   - 最终控制的是平均 hint 使用量，而不是某个死板上限。

2. `target average depth`
   - 不直接规定“不能超过 1”。
   - 而是通过一个超参把平均 hint depth 推到某个目标区间，例如 `0.8~1.0`。

3. `time-based hint budget`
   - 前期允许较深 hint，后期预算逐步收紧。
   - 这其实和上面的 UFT-style curriculum 可以统一起来。

4. `teacher-map + shallow prior`
   - 仍然用 task-aware fixed map 作为 teacher；
   - 但让最终使用的 depth 在它附近被一个“偏浅 prior”拉住，而不是直接照单全收。

### 4.2 当前最值得试的不是“max hint 1”，而是“matched budget”

最值得做的第一个实验，不是：

- `max_hint_depth = 1`

而是：

- 让 corrected fixed 和 dynamic / old fixed 在“平均 hint depth”上尽量匹配，然后比较谁更好。

这是因为我们现在其实还没有把两个因素拆开：

1. 深浅本身的影响。
2. 稳定性 / gate / schedule 的影响。

如果平均 depth 不匹配，就很难知道 dynamic 输 fixed，到底是因为：

- 它更抖；
- 还是因为它实际上给得更深或更浅。

### 4.3 这条线的推荐实验顺序

1. 先做 fixed family 内部的 shallow-budget sweep。
2. 再做和 dynamic 的 matched-depth 对照。
3. 最后才考虑是否需要显式 hard cap。

如果 shallow-budget fixed 已经能吃到 old fixed 的大部分收益，那比“直接强行 max hint 1”要干净得多。

## 5. 探索主线 D：为什么 dynamic 目前比 fixed 差

这个问题现在其实已经有一个比较清楚的主假设，但还没有被彻底验证。

deepresearch 里一个很重要的总结是：

> fixed 赢 dynamic，更像是“稳定但略偏的 scaffold”打赢了“更自适应但 conditioning schedule 更不平滑的 scaffold”。

见：

- [genrec_rl_study_2026-03-28.tex](/Users/fanghaotian/Desktop/src/GenRec/docs/deepresearch/genrec_rl_study_2026-03-28/genrec_rl_study_2026-03-28.tex#L540)

### 5.1 当前最可信的几个原因

#### 原因 1：dynamic 现在选的是 minimum hit depth，不是 best training depth

当前 dynamic 的 gate 本质上是：

- 只要某个 stage 的 group 里有任意一条 completion exact hit，就停在这一层。

这意味着它优化的是：

- “当前策略最少要给几层 hint，才至少能打中 1 条”

而不是：

- “哪一层 hint 最能形成稳定、可迁移的训练分布”

这两者不等价。

#### 原因 2：group-level one-hit gate 天然带噪声

dynamic 的 stopping signal 不是看这组 completion 整体质量，而是看：

- `rule_hit_any`

这会导致一个问题：

- 一个 group 里只要 16 条里有 1 条刚好中，整个 group 就在浅层停下；
- 但剩下大多数 completion 可能仍然很差。

这会让 selected stage 更像“碰巧命中”，而不是“这一层已经整体可学”。

#### 原因 3：conditioning schedule 不稳定

fixed 的 prompt distribution 是稳定的。

dynamic 的 prompt distribution 会随着训练推进改变：

- 早期更多样本要到深层才解开；
- 后期越来越多样本在浅层就停；
- 同一个样本 across steps 也可能在不同深度来回跳。

这会让 dynamic 的训练分布非平稳。

在当前只有 `2 epoch` 的训练强度下，这种 non-stationarity 可能直接吃掉了它理论上的自适应优势。

#### 原因 4：old fixed 其实踩中了“更浅但更稳”的甜点

这也是为什么现在不能简单说：

- “dynamic 理论上该赢 fixed，所以一定是实现还不够好”

更合理的说法是：

1. hint 少一点，可能确实更接近最终 no-hint eval；
2. 但 hint 深度又最好稳定；
3. old fixed 恰好是“更浅但更稳”；
4. dynamic 则更像“可能更浅，但不够稳，而且 gate 也不对”。

### 5.2 如果要让 dynamic 真正成为主线，我认为优先改的是 gate

这里优先级最高的不是再加一个 reward form，而是直接改 stopping signal。

可以考虑的候选包括：

1. 用 `rule_hit_count` 代替 `rule_hit_any`
2. 用 group mean reward 代替 any-hit
3. 用 prefix progress / matched prefix length 代替 binary exact hit
4. 给 dynamic depth 加 hysteresis / EMA，减少同一样本的深度抖动
5. 用 offline teacher map 作为 prior，而不是完全让 rollout 当步决定 depth

一句话说：

> dynamic 最大的问题不是“它会变”，而是“它现在用错了变的依据”。

## 6. 当前建议的执行顺序

为了避免同时开太多线，建议按下面顺序推进。

### 第一优先级：先把观测量补全

先补日志，不然很多争论只会停留在猜测。

建议优先补：

1. fixed run 的 `oracle_hint_depth_mean`
2. fixed run 的 hint depth 分 task breakdown
3. dynamic run 的 depth volatility 指标
4. `mean_length + mean_hint_depth` 的联合图

### 第二优先级：做 fixed + curriculum，而不是先折腾更多 reward

最值得先做的主实验：

1. `fixedhint-taskfix-b16-sid-only`
2. `fixedhint-taskfix-b16-sid-only + UFT-style anneal`
3. `fixedhint-taskfix-b16-sid-only + UFT-style anneal + hint SFT loss`

### 第三优先级：做 shallow-budget 对照

在 corrected fixed family 内部先回答：

- 更浅一些，到底能不能逼近 old fixed 的 top-10 优势？

### 第四优先级：再去重做 dynamic

到这一步再去改 dynamic gate，会更容易判断：

- 是 dynamic 本身不行；
- 还是只是 gate 错了；
- 还是只是 average depth 没匹配。

## 7. 当前不建议优先做的事

下面这些方向不是不能做，而是现阶段优先级不高：

1. 再扩一批 prefix reward 细变体。
2. 直接把 `max_hint_depth` 硬砍到 `1`，但不记录 budget 和 depth 分布。
3. 在还没对齐 average depth 之前，就直接下结论说 dynamic 不如 fixed。
4. 继续把 old fixed 当成“正确方法”，而不把它当作“浅而稳的现象学上界”。

## 8. 一句话版路线图

如果只保留一句话，后面这段最值得一直提醒自己：

> 现在最应该做的是：先把 hint 深度、completion length 和 transfer gap 之间的关系量化清楚；然后用 UFT-style curriculum 或 budgeted hint 去显式复现 old fixed 那种“更浅但更稳”的优势；最后再回头重做 dynamic gate，而不是继续盲目扩 reward shaping。
