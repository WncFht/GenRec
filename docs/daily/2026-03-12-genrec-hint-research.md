# GenRec Hint 方案调研笔记（2026-03-12）

本文档聚焦于 4 篇与你当前问题最相关的论文，并结合 `GenRec` 现有 RL pipeline，回答一个核心问题：

> 当 `GenRec` 的 RL 训练里约 80% 的 sample 在 `rule_only` 下拿不到 advantage 时，应该如何设计更合适的 hint / scaffold / progress-signal 机制？

## 1. 当前问题的本质

`GenRec` 当前的 RL 配置并不是 NuRL 那种“自然语言推理 + 多次随机 rollout + pass rate trigger”的场景，而是：

- 使用 **constrained beam search** 生成 SID 候选，见 `GenRec/MIMIGenRec.py`、`GenRec/util.py`。
- 当前主跑脚本默认是 `rule_only`，即只有 **exact match** 才有 `1.0` reward，否则 `0.0`，见 `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh`。
- beam 是相关候选，不是 i.i.d. rollout，因此“pass rate”并不是最自然的难度指标。
- 由于 exact reward 极稀疏，当某个 prompt 的整组 beam 都 miss 时，这一组基本没有有效 advantage。

因此，对 `GenRec` 更贴切的问题表述应该是：

1. 如何把“没有 exact hit 的 beam 组”变成**仍有训练信号**的组？
2. 如何在必要时给 **结构化 hint**，而不是 NuRL 式文本 hint？
3. 如何避免模型只是**抄 hint prefix**，而没有真正学会后续 token 判别？

---

## 2. 这 4 篇论文分别能提供什么

## 2.1 ADHint

- 论文：`https://arxiv.org/abs/2512.13095`
- 本地源码包：`paper_research/ADHint_paper`
- 公开代码：这次检索里没有找到明确的官方公开仓库

### 核心机制

ADHint 不是简单地“给 hint”，而是把 difficulty 引入了 hint 训练的两个关键位置：

1. **Adaptive Hint with Sample Difficulty Prior (AH-SDP)**
   - 先跑 naive-rollouts。
   - 用 naive-rollouts 的平均 reward 估计 sample difficulty。
   - 再据此决定 hint ratio。
   - 公式核心是：
     - `Diff_N = 1 - mean(r_1, ..., r_n)`
     - `w = (w_max - w_min) * Diff_N + w_min + noise`
   - 直觉：越难的 sample，hint 越长；越简单的 sample，不需要 hint。

2. **Consistency-based Gradient Modulation (CGM)**
   - 对 hint token 不做一刀切学习，而是比较：
     - hint token 的 entropy
     - continuation 的平均 entropy
   - 如果 hint token 的分布和模型自身 continuation 分布相差太大，就下调这些 token 的梯度。
   - 核心思想：避免 off-policy hint 把 policy 硬拽向 teacher distribution。

3. **Selective Masking for Hint Preservation**
   - 如果 hint-rollout 最终 advantage 为负，那么不要把这个负梯度回传到 hint token 上。
   - 原因很直接：hint prefix 本身是假定正确的，失败的是 continuation，不应该惩罚提示本身。

4. **Advantage Estimation with Rollout Difficulty Posterior (AE-RDP)**
   - 不把 naive-rollouts 和 hint-rollouts 简单混成一组做标准 advantage。
   - 而是显式利用两边 difficulty 的相对关系，对 advantage 再加权。
   - 直觉：
     - 更难的 naive positive rollout 更有价值
     - 更容易的 hint negative rollout 应该罚得更重

### 对 GenRec 的最大启发

ADHint 对 `GenRec` 最有价值的不是“hint ratio”本身，而是 3 个结构性思想：

- **hint 强度必须跟 sample difficulty 绑定**
- **guided / unguided rollout 不能简单混成一组算 advantage**
- **hint token 不能像普通生成 token 一样无脑参与 RL 更新**

这三个点都非常贴近 `GenRec`。

---

## 2.2 SEELE

- 论文：`https://arxiv.org/abs/2509.06923`
- 本地论文源码：`paper_research/SEELE_paper`
- 本地代码：`paper_research/SEELE_code/seele-master`

### 核心机制

SEELE 的出发点是：

- 训练效率在 rollout accuracy 太低或太高时都不好；
- 最好的“学习甜点区”大约在 **50% accuracy** 附近。

于是它做了一个 **instance-level hint length prediction**：

1. 把 rollout sampling 分成多轮（multi-round）。
2. 每轮用一个 hinting rate `p^(i)` 生成一批结果。
3. 收集 `(hint_rate, accuracy)` 对。
4. 用一个 IRT 风格的 3PL 曲线 `f_phi(p)` 拟合 accuracy-hint 关系。
5. 反解出“为了达到目标 accuracy（如 50%），当前 sample 应该给多长 hint”。

### 训练层面的关键点

SEELE 很重要的一点是：

- **只在生成出的 continuation token 上计算 RL loss**
- 对 hint token 额外加监督项，而不是把 response-level advantage 直接分发到 hint token 上

论文里明确指出：

- 如果 response-level advantage 直接分给 hint token，当模型在有 hint 的前提下仍然失败，会错误地压低模型对 hint 的概率。

代码里也能看到它显式构造了 `hint_mask`：

- `paper_research/SEELE_code/seele-master/seele/seele_dataset.py`
- `paper_research/SEELE_code/seele-master/seele/seele_ray_trainer.py`
- `paper_research/SEELE_code/seele-master/seele/hint_manager.py`

### 对 GenRec 的最大启发

SEELE 的最大价值在于两个点：

1. **hint 长度应当实例级自适应，而不是全局固定**
2. **hint token 与 continuation token 应该分开处理损失**

但 SEELE 的完整 IRT + multi-round online 拟合，在 `GenRec` 里不一定要照搬。

原因是：

- `GenRec` 的 SID 通常长度非常短（比如 4 个 token）
- hint 强度本质上只是 `{0, 1, 2, 3}` 这样的离散深度选择问题
- 因此比起连续 hint ratio 回归，更自然的是一个**离散 hint depth 调度器**

换句话说：

- SEELE 的“思想”非常有用；
- SEELE 的“完整 IRT machinery”对 `GenRec` 可能过重。

---

## 2.3 CoFiRec

- 论文：`https://arxiv.org/abs/2511.22707`
- 本地论文源码：`paper_research/CoFiRec_paper`
- 本地代码：`paper_research/CoFiRec_code/CoFiRec-main`

### 核心机制

CoFiRec 不是 hint 论文，但它对你非常重要，因为它把 generative recommendation 的 token 结构讲清楚了：

- item tokenization 应该是 **coarse-to-fine hierarchy**
- 前面的 token 承载高层语义（如 category）
- 后面的 token 承载更细粒度语义与 CF disambiguation

论文方法段明确写到：

- semantic hierarchy tokenized item: `S_i = [s_i^(1), ..., s_i^(K)]`
- 前 `K-1` 个 token 是 progressive semantic levels
- 最后一个 token 用于 CF information

也就是说，对这类结构化 SID：

- **前缀不是任意前缀**
- 它本身就有 coarse-to-fine 语义含义

### 对 GenRec 的最大启发

这几乎直接定义了 `GenRec` 的 hint 形式：

- 最自然的 hint 不是文本
- 而是 **暴露前 1 个 / 前 2 个 / 前 3 个 SID token**

所以如果以后你在 `GenRec` 做 hint：

- `h=1` 表示给 coarse semantic hint
- `h=2` 表示给更强一点的语义 hint
- `h=3` 表示接近“只剩最后判别”的强 scaffold

这比 NuRL 的自然语言 hint 贴场景得多。

---

## 2.4 Token-Weighted Multi-Target Learning

- 论文：`https://arxiv.org/abs/2601.17787`
- 本地论文源码：`paper_research/TokenWeightedMTL_paper`
- 本地代码：`paper_research/TokenWeightedMTL_code/Token-Weighted-Multi-Target-Learning-for-Generative-Recommenders-with-Curriculum-Learning-main`

### 核心机制

这篇论文说明了 generative recommendation 的一个本质：

- 不同 token 对最终 item identification 的贡献并不相同。

它提出两种 weighting：

1. **Front-Greater Weighting**
   - 强调前面 token 更重要
   - 理由：前缀 token 更能减少 candidate uncertainty
   - 换句话说，早期 token 的 conditional information gain 更高

2. **Frequency Weighting**
   - 强调稀有 token 更重要
   - 理由：长尾 token 更具区分性，能缓解 popularity bias

然后它把：

- front-greater loss
- frequency loss
- original CE loss

放到一个 **multi-target learning + curriculum learning** 框架里：

- 早期更看重 Front-Greater + 原始 CE
- 后期更看重 Frequency

代码里这件事也写得很直接：

- `paper_research/TokenWeightedMTL_code/.../TIGER/modeling_letter.py`

### 对 GenRec 的最大启发

这篇论文对 `GenRec` 的 hint 设计有两个强结论：

1. **早期 token 更值得被保护/引导**
   - 这为“hint 应优先作用在前 1~2 个 SID token 上”提供了非常强的支持

2. **训练目标本身也应该有 curriculum**
   - 早期：先把 coarse prefix 学稳
   - 后期：再转向 rare/fine token discrimination

这意味着：

- 即使你不显式做 hint，这篇论文也支持你先加强 prefix-level reward
- 如果你做 hint，这篇论文会支持你做 **front-heavy 的 hint policy**

---

## 3. 四篇论文放在一起，哪个最贴 `GenRec`

如果按“与 `GenRec` 的适配度”排序，我的判断是：

1. **CoFiRec**：定义了为什么 SID 本身就该 coarse-to-fine
2. **Token-Weighted**：定义了为什么前面 token 更重要，且应该 curriculum 化
3. **ADHint**：定义了如何做 adaptive hint schedule、selective masking、difficulty-aware advantage
4. **SEELE**：定义了实例级 hint-length 预测，但其连续 IRT machinery 在 `GenRec` 里可能过重

所以真正适合 `GenRec` 的，不是任何一篇单独照搬，而是一个组合：

- **CoFiRec + Token-Weighted** 负责告诉我们 hint 的“单位”和“方向”
- **ADHint + SEELE** 负责告诉我们 hint 的“调度”和“loss 设计”

---

## 4. 对 `GenRec` 当前 RL pipeline 的重新判断

## 4.1 当前痛点不只是“没 hint”

根据 `GenRec` 当前代码，真正的问题其实分两层：

1. **reward 太 sparse**
   - 当前主脚本默认 `rule_only`
   - exact match 才有正信号
   - 于是大量 beam 组全 0

2. **beam search 不是独立 rollout**
   - NuRL 那种 pass rate 直觉不一定成立
   - 4 个/16 个 beam 的“成功比例”不是 i.i.d. 估计，而是相关排序结果

因此，直接把 NuRL 的“根据 pass rate 调 hint”平移过来，很可能不稳。

## 4.2 更适合 `GenRec` 的难度指标

对 `GenRec`，我更推荐这些 difficulty / progress 指标：

- `m_max`：一组 beam 中 **最长正确 prefix 匹配长度**
- `m_mean`：一组 beam 的平均 prefix 匹配长度
- `any_exact`：是否已有 exact hit
- `beam_entropy`：同一位置上 beam 的分歧程度
- `ranked_prefix_score`：按 beam rank 加权的 prefix progress

这些量比“pass rate”更贴 beam-search setting。

---

## 5. 我对 `GenRec` 的推荐设计

下面是我目前最推荐的设计，不是最终实现，只是研究结论导出的方案草图。

## 5.1 先不要直接在 `rule_only` 上堆 hint

如果当前确实有 ~80% sample 拿不到 advantage，那么第一步不应该是先加 hint，而应该先把 **progress signal** 接起来。

我建议的顺序是：

### 第一步：把 reward 从 `rule_only` 提升到 prefix-aware

优先尝试：

- `prefix_rule_only`
- 或 `prefix_only`
- 并打开 token-level prefix advantage

理由：

- 这一步是最符合 `GenRec` 现有代码结构的
- 风险最低
- 能先把“80% 全 0”这个问题压下去

换句话说：

- 在 `GenRec` 里，**prefix reward 本身就是一种 process reward**
- 很多情况下，它比文本 hint 更自然

## 5.2 如果要做 hint，hint 应该是 **SID prefix scaffold**

具体来说：

- `h=0`：不给 hint
- `h=1`：暴露第 1 个 SID token
- `h=2`：暴露前 2 个 SID token
- `h=3`：暴露前 3 个 SID token

这就是 `GenRec` 的“hint 强度”。

它和 NuRL 的区别是：

- NuRL hint 是文本概念指导
- `GenRec` hint 是结构化 target prefix

## 5.3 hint 强度不该按 pass rate 调，而该按 prefix progress 调

推荐的 sample difficulty prior：

```text
s_prefix = max_prefix_match / L
s_rank   = rank-weighted average prefix progress
d        = 1 - s_prefix
```

一个简单初版规则可以是：

- 如果 `any_exact = 1`：`h = 0`
- 如果 `max_prefix_match = 0`：`h = 2`
- 如果 `max_prefix_match = 1`：`h = 1`
- 如果 `max_prefix_match >= 2`：`h = 0`

后续再把它连续化，做成：

- `h = scheduler(d, beam_entropy, stage)`

## 5.4 不要把 guided 和 unguided beam 混成一锅算 advantage

这点 ADHint 非常关键。

如果以后你做：

- 一部分 beam 不带 hint
- 一部分 beam 带 `h=1/2` hint

那么这两类 beam 的难度明显不同，直接一起做 group-normalized advantage 会有偏差。

我建议：

- 先分别计算 naive group 和 guided group 的 group statistics
- 再做 difficulty-aware reweighting

即便不完整照抄 ADHint 的公式，也至少要保证：

- **naive positive beam 的优势不要被 guided positive beam 淹没**
- **guided negative beam 应该比 naive negative beam 罚得更重**

## 5.5 hint token 必须做 selective masking

这点我认为是必须的。

原因是：

- 对 `GenRec` 而言，hint token 是 target SID 的真前缀
- 如果 guided beam 最终失败，失败原因几乎总是在 continuation，而不是 hint prefix 本身
- 所以不能把负 advantage 回传到 hint prefix token 上

更进一步，我甚至建议：

- **hint token 默认就不参加 RL policy loss**
- 只对 hint 后面的 continuation 做 RL 更新
- 如有需要，再给 hint token 一个很小的 CE / imitation loss

这比 ADHint 更激进，但我认为更适合 recommendation SID 场景。

原因是：

- recommendation 的 prefix hint 比自然语言 hint 更接近“标准答案片段”
- 更容易诱发抄前缀而不学 continuation 的问题

## 5.6 SEELE 的 IRT 思想可以保留，但没必要原样搬

我不建议在 `GenRec` 里直接上：

- multi-round rollout
- per-instance 3PL 拟合
- online inverse regression

因为你的 action space 很短：

- 如果 SID 长度就是 4，那么 hint depth 也只有 4 个离散档

更合适的替代是：

- 做一个 **discrete hint-depth bandit / lookup table**
- 维护每个 difficulty bucket 下 `{h=0,1,2,3}` 的成功率或 prefix-progress 改善
- 让系统逐步学会“当前这类样本给多少 hint 最合适”

这相当于把 SEELE 的连续 hint-rate 预测，离散化成 `GenRec` 能承受的版本。

## 5.7 curriculum 也应该做成前重后轻

来自 Token-Weighted 的启发，我建议整个 RL/hint pipeline 带上阶段性调度：

### 训练前期

- prefix reward 权重大
- `h_max = 2`
- 更重视 coarse token 学稳

### 训练中期

- `h_max = 1`
- guided beam 数减少
- 提高 suffix discrimination / rank reward 的权重

### 训练后期

- 尽量 `h = 0`
- 只对最难样本保留极弱 prefix scaffold
- 更多依赖真实无 hint 生成能力

---

## 6. 一个比较贴 `GenRec` 的最终 pipeline 草图

下面是我当前最推荐的形态。

### Phase A：先解决 80% 无 advantage

- reward 从 `rule_only` 改为 prefix-aware
- 打开 token-level prefix advantage
- exact rule reward 作为 probe / eval 指标保留

### Phase B：引入 adaptive prefix hints

对每个 prompt：

1. 先生成 unguided beams
2. 计算 `max_prefix_match / mean_prefix_match / beam_entropy`
3. 由 scheduler 决定 hint depth `h`
4. 再生成 guided beams（只对部分样本、部分 beam）

### Phase C：difficulty-aware advantage

- unguided 和 guided 分开统计
- 再做 reweighting
- 避免 guided beam 主导整个更新方向

### Phase D：selective masking

- hint token 在负优势 guided beam 上全部 mask
- 更激进版本：hint token 在所有 RL loss 中都 mask，只让 continuation 学 RL

### Phase E：front-heavy curriculum

- 前期更重前缀
- 后期更重 rare/fine token
- hint depth 逐步退火

---

## 7. 我目前最明确的研究判断

如果要做一个“比 NuRL 更新、更贴 `GenRec`”的方案，我不建议你做：

- 文本 hint
- all-fail binary trigger
- 统一 hint ratio
- guided / unguided 混组标准 GRPO

我建议你做的是：

- **SID prefix scaffold** 作为 hint
- **prefix-progress / uncertainty** 作为 difficulty signal
- **difficulty-aware guided/unguided advantage**
- **selective masking / no-RL-on-hint-tokens**
- **front-heavy curriculum**

一句话总结就是：

> 对 `GenRec`，最自然的现代化 hint 方案不是 NuRL 风格的文本 hint，而是 **adaptive prefix scaffolding with selective masking and prefix-progress-aware credit assignment**。

---

## 8. 本地参考材料位置

这次下载到本地的论文/代码如下：

- `paper_research/ADHint_paper`
- `paper_research/SEELE_paper`
- `paper_research/SEELE_code/seele-master`
- `paper_research/CoFiRec_paper`
- `paper_research/CoFiRec_code/CoFiRec-main`
- `paper_research/TokenWeightedMTL_paper`
- `paper_research/TokenWeightedMTL_code/Token-Weighted-Multi-Target-Learning-for-Generative-Recommenders-with-Curriculum-Learning-main`

---

## 9. 下一步建议

我建议后续工作按下面顺序推进：

1. 先把 `GenRec` 当前 `rule_only` 改成可观测 prefix signal 的研究 baseline
2. 再设计 `hint depth scheduler`
3. 再设计 `guided/unguided advantage` 和 `hint token masking`
4. 最后再决定要不要做 SEELE 风格的 epoch-level adaptive memory

如果直接一步到位做完整 adaptive hint system，调试成本会非常高。

---

## 10. 固定 `16 rollout` 约束下的 V1 Hint 策略（2026-03-13 更新）

这一节是在前文研究结论基础上，专门针对当前 `GenRec` 实验约束收敛出来的 **V1 落地方案**。

当前明确约束是：

1. 为了和既有实验对齐，**每个 sample 总 rollout 数必须固定为 `16`**。
2. **不改变 guided / unguided 的比例**，也就是说不同 sample 之间不要动态改成 `12+4`、`4+12` 之类。
3. V1 不希望过于复杂，因此不引入 SEELE 式的 multi-round IRT 拟合，也不引入样本级历史搜索策略。
4. 当前主场景仍然基于 **beam search**，而不是采样 rollout。

在这些约束下，我认为最合理的 V1 不是复杂的 fully adaptive pipeline，而是一个**固定组比 + 弱自适应 hint 深度**的双组方案。

### 10.1 结论先说

我推荐的 V1 设计是：

- **每个 sample 固定 `16` 个 rollout**。
- **固定拆成 `8 unguided + 8 guided`**。
- **两组都使用 beam search**。
- guided 组的 hint 深度只在 `h ∈ {1, 2}` 中选择，不使用 `h=0` 和 `h=3`。
- `h` 的决定方式只看 **unguided 组是否出现任何 prefix progress**，不做更复杂的 per-sample 历史建模：
  - 如果 unguided 组里 **至少有一个 beam 拿到正的 prefix reward**，则 guided 组用 **`h = 1`**。
  - 如果 unguided 组里 **一个 beam 都没有拿到 prefix reward**，则 guided 组用 **`h = 2`**。
- **hint token 在 RL loss 中全部 mask**，只对 hint 后的 continuation 计算 reward / advantage / backward。
- guided 组和 unguided 组的 advantage **分开做组内归一化**，最后再拼回一个总 batch。
- 如果只跑 `2 epoch`，可以**记录历史 rollout 统计**，但 **V1 不把历史统计作为 hint 调度输入**。

这一版的目标不是“最优”，而是：

- 不破坏和既有 `16 rollout` 实验的可比性；
- 尽量少改系统结构；
- 同时缓解当前 exact / prefix reward 稀疏导致的大量全零组问题。

---

### 10.2 为什么是固定 `8 unguided + 8 guided`

在当前约束下，`8 + 8` 是最稳妥的选择。

原因如下：

1. **结构简单**
   - 一个 sample 先生成 `8` 个不带 hint 的 beam。
   - 再生成 `8` 个带 hint 的 beam。
   - 总数固定为 `16`，组比固定，不会破坏与旧实验的对齐。

2. **可以保留 unguided frontier 信号**
   - 如果 `16` 个 beam 全改成 guided，模型当前真实能力边界就看不见了。
   - 保留 `8` 个 unguided beam，至少还能观察：
     - 当前 sample 是否有 exact hit
     - 是否有 prefix progress
     - 当前最远前缀能走到哪里

3. **guided 组数量足够形成稳定子组**
   - 如果 guided 只有 `4` 个 beam，那么在很多 hard sample 上方差会较大；
   - `8` 个 guided beam 更适合作为独立 group 做组内归一化。

4. **实现复杂度相对可控**
   - 当前 `GenRec` 的 trainer 假设每个 sample 有一组固定 `num_generations` 的 completion；
   - 改成固定双组（`8+8`）比改成动态比例更容易落地。

换句话说：

- `8+8` 不是因为理论上一定最优；
- 而是因为在“固定 16 + 不变比例 + 不想太复杂”这三个条件一起成立时，它是最合理的 V1 平衡点。

---

### 10.3 为什么两组都继续用 beam search

我建议 V1 里：

- unguided 组：beam search
- guided 组：beam search

不要一边 beam、一边 sampling。

原因：

1. 当前 `GenRec` 的 reward、group normalization、constrained decoding 都是围绕 beam 设计的。
2. 如果一边用 beam、一边用 sampling，guided / unguided 两组的难度和分布来源会同时变化，不利于分析。
3. 你当前真正想解决的是“全零 advantage 组”，不是“增加 rollout 多样性”本身。

因此，V1 最好只改 **conditioning（是否加 prefix hint）**，不要同时改 **decoding policy**。

---

### 10.4 hint 深度为什么只取 `1` 或 `2`

在 `GenRec` 这里，如果目标 SID 长度是 `4`，理论上 hint 深度可以取：

- `h=0`：无 hint
- `h=1`
- `h=2`
- `h=3`

但在固定 `8+8` 的 V1 里，我不建议使用 `h=0` 和 `h=3`：

#### 不建议 `h=0`

如果 guided 组也取 `h=0`：

- 它和 unguided 组在输入上没有区别；
- 在 beam search + 相同 prompt 的情况下，guided 组很可能只是 unguided 组的重复；
- 这会浪费一半 rollout 预算。

因此，只要我们保留固定 guided 组，就应确保 guided 组与 unguided 组**语义上不同**。

#### 不建议 `h=3`

如果 `h=3`：

- 对 4-token SID 来说，只剩最后一个 token 需要模型生成；
- 这太接近“把答案前缀直接给了模型”；
- continuation 学习信号太短，容易变成“只训练最后一位分类器”，而不是帮助整体 RL 探索。

因此，V1 里更合理的 scaffold 强度是：

- `h=1`：弱 scaffold，只给最粗粒度语义前缀
- `h=2`：强 scaffold，给到中等粒度的结构前缀

这也是为什么我推荐 V1 只在 `{1, 2}` 上做选择。

---

### 10.5 V1 的 hint 深度调度规则

这是当前我最推荐的 **最简自适应规则**：

#### Step 1：先生成 `8` 个 unguided beams

对每个 beam 计算 longest matched prefix length，记为 `m_b`。

然后只取一个最简单的 sample-level 判据：

```text
prefix_hit_any = 是否存在某个 beam 满足 m_b > 0
```

也就是：

- 只要有任何一个 unguided beam 至少命中 1 个 prefix token，说明这个 sample 已经不是“完全不会”；
- 如果 8 个 beam 全都 `m_b = 0`，说明这个 sample 对当前 policy 来说是“all-zero prefix frontier”。

#### Step 2：根据 `prefix_hit_any` 决定 guided 组 hint 深度

规则如下：

```text
if prefix_hit_any:
    h = 1
else:
    h = 2
```

#### 直觉解释

- **有 prefix 命中 -> `h=1`**
  - 模型至少已经能进入正确子树的边缘；
  - 这时只给一个 coarse token 的 hint，就足够帮助 guided 组把注意力放到后续 suffix 判别上；
  - 不需要给太强的 scaffold。

- **完全没有 prefix 命中 -> `h=2`**
  - 模型当前连第一个正确前缀都走不出来；
  - 如果只给 `h=1`，guided 组仍然可能过难；
  - 给 `h=2` 能更稳定地产生可学习的 continuation 轨迹。

#### 为什么不用更复杂的 `m_max / m_mean / beam_entropy` 调度

并不是这些指标不重要，而是：

- 你现在明确不希望 V1 太复杂；
- 且当前 prefix reward 本身已经比 exact reward 稠密，但依然比较稀疏；
- 因此 V1 最好先用一个**二段式简单规则**，把 hardest-all-zero 和 not-all-zero 区分开。

这条规则已经足以回答：

> “应该 hint 几个 token？”

在当前 V1 里，答案就是：

- **完全没有 prefix frontier 的 sample -> hint 2 个 token**
- **已经存在 prefix frontier 的 sample -> hint 1 个 token**

这条规则和前文的 CoFiRec / Token-Weighted 结论也一致：

- hint 优先作用在前面 token 上；
- 但不要一上来给太深，避免把问题退化成只猜最后一个 token。

---

### 10.6 为什么不用“exact hit / rule reward 命中数”来判 hint 深度

因为在 beam-search setting 下，`rule_reward` 太粗。

一个 sample 出现 `0` 个 exact hit，可能对应完全不同的两种情况：

1. 连第一个 token 都走不对；
2. 前 3 个 token 都走对了，只差最后 1 个 token。

如果都按照 “0 exact -> 强 hint” 处理，就会把很多“其实已经有明显 prefix frontier”的 sample 也过度简化。

因此，在 `GenRec` 里更合理的是：

- `rule_reward` 用来判断是否最终 exact 命中；
- `prefix frontier` 用来判断**当前 sample 离正确路径还有多远**；
- hint 深度应该主要由后者决定。

---

### 10.7 guided 组的 reward 应该怎么算

这点非常关键。

如果 guided 组使用 `h=1` 或 `h=2` 的真实 target prefix，而 reward 仍然从第一个 token 开始计算，那么 guided 组会天然拿到一部分“白给”的 prefix reward。

这会产生两个问题：

1. guided 组的分数被虚高；
2. 模型学到的是“带着答案前缀时当然更容易”，而不是 continuation 的真实能力。

因此，**guided 组的 reward 也必须做 prefix masking**。

### V1 的 guided reward 规则

假设 hint 深度是 `h`：

- 对 guided 组：
  - `rule_reward` 仍看完整 SID 是否最终 exact hit；
  - **prefix reward 只从第 `h+1` 个 token 开始计算**；
  - 也就是说，被 hint 的前 `h` 个 token 不计 reward。

例如目标是 `[a, b, c, d]`：

- 若 `h=2`，guided prompt 里已经给了 `[a, b]`
- guided completion 如果生成成 `[a, b, x, y]`
  - 前 2 个 token 不能算 prefix reward
  - 只从 `[c, d]` 的 continuation 对齐情况算 guided prefix reward

这样 guided 组才真正反映：

- “在给定 scaffold 之后，模型有没有学会后续 token 的判别”

---

### 10.8 hint token 要不要 mask：V1 的明确结论

这里我的结论非常明确：

> **要 mask，而且 V1 直接把 hint token 全部从 RL loss 中 mask 掉。**

也就是：

- unguided 组：正常计算所有生成 token 的 RL loss
- guided 组：
  - hint token：`RL mask = 0`
  - continuation token：正常计算 reward / advantage / backward

### 为什么要这么做

1. 对 `GenRec` 而言，hint token 不是自然语言概念提示，而是**真实 target SID 的前缀**。
2. 这些 token 更像“conditioning context”，而不是 policy 自己应当负责的 action。
3. 如果 guided beam 最终失败，失败几乎总发生在 continuation，而不是 hint prefix 本身。
4. 因此，不能把负 advantage 反传到 hint token 上。

### 为什么 V1 直接“全 mask”，而不是 ADHint 那种 selective masking

ADHint 的 selective masking 更适合文本 reasoning hint：

- hint token 仍可能作为 teacher distribution 的一部分被部分学习；
- 只有在负优势时才 mask。

但在 `GenRec` 这里：

- hint token 是结构化 target prefix；
- 它和最终答案的耦合比文本 hint 强得多；
- 因此更保守、更适合 recommendation SID 场景的做法是：
  - **hint token 默认全部不参与 RL loss**；
  - 如果未来需要，再单独给 hint token 一个小的 imitation / CE loss。

所以 V1 的建议是：

- **guided prefix token: 全 mask**
- **guided continuation token: 正常 RL**

---

### 10.9 guided / unguided advantage 应该怎么处理

我不建议把 `8 unguided + 8 guided` 简单拼成一个 `16` 大组后直接做统一 group normalization。

原因：

- guided 组显式更容易；
- 如果混成一组，guided 正样本很容易主导整个 advantage 分布；
- 最后模型学到的是“如何在有 scaffold 时补完 suffix”，而不是提升 unguided 能力。

### V1 的建议

- unguided 组内部做一次 group normalization（基于 8 个 unguided beams）
- guided 组内部做一次 group normalization（基于 8 个 guided beams）
- 然后把两组优势拼接回一个总 batch

这样做的好处是：

- 两组各自内部有可比性；
- guided 组不会直接淹没 unguided 组；
- 也不需要一上来就实现更复杂的 ADHint-style difficulty posterior correction。

换句话说：

- V1 先做到“**分组归一化**”；
- V2 再考虑“**跨组 difficulty-aware reweighting**”。

---

### 10.10 要不要记录前一轮 / 前一 epoch 的 rollout 情况

**要记录，但 V1 不依赖这些历史来决定 hint。**

如果只跑 `2 epoch`，我不建议像 SEELE 那样做历史驱动的调度器。

原因：

1. 历史太短，sample-level 统计很容易不稳定；
2. V1 的目标是先验证固定 `16 rollout` + prefix scaffold 是否有效；
3. 把历史闭环接进调度会显著增加系统复杂度。

### V1 建议记录的统计量

建议每个 sample 至少记录：

- unguided 组：
  - `u_any_exact`
  - `u_any_prefix`
  - `u_m_max`
  - `u_m_mean`
- guided 组：
  - `g_any_exact`
  - `g_suffix_prefix_hit`
  - `g_suffix_prefix_mean`
- 调度量：
  - 当前 step 选择的 `h`

### 这些历史怎么用

- **V1**：只用于日志分析，不参与决策。
- **V2**：可以考虑做 EMA 或 bucket-level scheduler，而不是 sample-level 直接查表。

因此，对你当前“只跑 2 epoch”的设置，我的明确建议是：

- **记录历史**：要
- **把历史接入 V1 hint 决策**：不要

---

### 10.11 V1 最终流程（推荐版）

对每个 sample：

1. 用原始 prompt 跑 **8 个 unguided beam**。
2. 根据 unguided 组判断：
   - 是否存在任何 prefix hit。
3. 决定 guided 组 hint 深度：
   - 有 prefix hit -> `h=1`
   - 无 prefix hit -> `h=2`
4. 在 prompt 中拼接对应深度的 SID prefix scaffold。
5. 再跑 **8 个 guided beam**。
6. 计算 reward：
   - unguided 组：正常 exact / prefix reward
   - guided 组：prefix reward 只在 hint 之后的 suffix 上计算
7. 计算 loss：
   - unguided：正常 RL loss
   - guided：hint token 全 mask，只对 continuation 做 RL
8. 计算 advantage：
   - 先各自在 8-beam 子组内做归一化
   - 再把两组结果拼回总 batch

这样，V1 就具备了 4 个非常重要的特性：

- 总 rollout 数与旧实验严格对齐：`16`
- guided / unguided 比例固定：`8+8`
- hint 强度足够简单：`h ∈ {1,2}`
- hint token 不会污染 RL credit assignment

---

### 10.12 对你原始想法的直接回答

你原来的想法是：

- 先不加 hint rollout 一组；
- 如果有 prefix reward 命中，再给 hint rollout 一组；
- 如果一个都没有，就给一个 hint 再 rollout 一组；
- 两组总 rollout 个数仍然是 16。

我认为这个方向是对的，但在你现在的约束下，应该改成下面这个更稳定的版本：

- **始终先跑一组 unguided**（固定 8 个）
- **始终再跑一组 guided**（固定 8 个）
- guided 组的区别不是“有没有”，而是“hint 深度是 1 还是 2”
- 不要把“是否开 guided 组”本身做成动态开关

原因是：

- 如果你把 guided 组做成有时存在、有时不存在，那么虽然 nominal rollout 还是 16，但实际逻辑会更复杂；
- 如果 guided 组不存在，还要考虑怎么补足另外 8 个 beam；
- 在 beam search 下，补成额外 unguided beam 又容易重复。

因此，对 V1 而言：

- **guided 组永远存在**
- **动态的只是 hint 深度，不是 guided 组是否存在**

这是当前约束下最合理的折中。

### 10.13 这套 V1 为什么不能只改 config，而需要定制 trainer

这一点也需要在设计阶段说清楚。

当前 `GenRec` 代码默认假设：

- 每个 sample 只有一组固定大小的 `num_generations` completion；
- 这一组 completion 对应同一种 prompt 形式；
- advantage 直接按这整个 group 做归一化。

因此，下面这些逻辑并不是改脚本参数就能得到的：

- `8 unguided + 8 guided` 双组生成
- guided 组根据 unguided 组结果决定 hint 深度
- guided / unguided 分组归一化
- guided reward 的 hint-prefix masking
- guided RL loss 的 hint-token masking

也就是说，这个 V1 的实现含义其实是：

1. **生成阶段要拆成两次**
   - 第一次：原始 prompt，生成 `8` 个 unguided beams
   - 第二次：带 prefix scaffold 的 prompt，生成 `8` 个 guided beams

2. **reward 阶段要区分两组**
   - unguided：正常 exact / prefix
   - guided：prefix reward 从 hint 后开始算

3. **advantage 阶段要区分两组**
   - 不能直接把 `16` 个 beams 混成一个普通 GRPO group

4. **loss 阶段要区分 token mask**
   - guided 组前 `h` 个 hint token 不参与 RL loss

所以，从工程上说，V1 更像是：

- 保留现有 `GenRec` reward / prefix signal 的大框架；
- 但在 trainer 里加入一个 **dual-group beam rollout path**。

这也是为什么我前面一直强调：

- 当前阶段先把设计定清楚；
- 不要误以为这是一个“只改 reward_mode 或 num_beams”的小改动。
