# GenRec 节点级 Rollout 失效分析（2026-03-17）

这份文档聚焦一个更具体的问题：

- 为什么这个理论上是 `256 x 4` 的 SID tree，在实际 rollout 时，前几个 token 一旦不给 hint，就很容易整条路径都 rollout 不出来？

本次分析基于本地 bundle：

- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle`

并新增了节点级分析脚本：

- [analyze_local_rollout_nodes.py](/Users/fanghaotian/Desktop/src/GenRec/scripts/hint_research/analyze_local_rollout_nodes.py)

对应新输出：

- [rollout_node_summary.json](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/rollout_node_summary.json)
- [rollout_node_stats.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/rollout_node_stats.csv)
- [rollout_parent_stats.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/rollout_parent_stats.csv)
- [rollout_feature_bin_summary.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/rollout_feature_bin_summary.csv)

## 一句话结论

最核心的发现不是“树太深”，而是：

1. 这棵树在前两层的实际使用形状和 `256 x 4` 的想象完全不一样。
2. 真正的 bottleneck 不在最后两层，而是在前两层，尤其是 depth-2 parent 的严重过载。
3. `base` 失败比较分散，是一个广泛的“第一跳路由问题”。
4. 一旦进入 `hint_1`，困难开始被少数高分支数 parent 主导。
5. 到 `hint_2 -> hint_3+`，失败已经快速收缩到少数 depth-3 prefix。

换成更直白的话：

- 第一个 token 没选对，后面整条 rollout 很容易跟着跑偏。
- 即使第一个 token 选对了，第二层很多 parent 下面仍然塞了几十到上百个 child，局部竞争极强。
- 所以“不给前几个 token hint 就 rollout 不出来”不是偶然，而是树形编码和当前模型能力在前两层发生了结构性失配。

## 1. 这棵树实际长什么样

理论上是 `256 x 4`，但这份 bundle 里的实际活跃结构是：

| 深度 | 活跃 token 数 | 活跃 parent 数 | mean sibling | median sibling | max sibling |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth 1 | 41 | 1 | 41.0 | 41.0 | 41 |
| depth 2 | 214 | 41 | 56.8 | 62.0 | 105 |
| depth 3 | 251 | 2329 | 2.32 | 1.0 | 22 |
| depth 4 | 256 | 5394 | 1.16 | 1.0 | 21 |

这组数最重要的含义是：

- 根节点并没有展开成 256 个 child，只用了 `41` 个 root token。
- 真正最拥挤的是 depth 2：
  - 41 个 root child 往下，平均每个 parent 有 `56.8` 个 children
  - 中位数 `62`
  - 最大 `105`
- 到 depth 3 以后，树突然就稀疏很多了：
  - 大多数 parent 只有 `1` 个 child
- 所以 rollout 的主要结构压力不是“树深 4 层”，而是“第二层局部选择空间太大”。

从 [rollout_node_summary.json](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/rollout_node_summary.json) 还能看到：

- depth-2 parent 的 median `parent_maxshare` 只有 `0.0776`
- depth-2 parent 的 mean entropy 约 `0.862`

这说明在第二层，大多数 parent 下面：

- 不存在明显 dominant child
- 大量 child 的相对占比都差不多

这恰恰是 rollout 最怕的情况。

## 2. rollout 失败主要卡在哪一层

从 bundle 的 transition rate 看：

- `base -> need_hint`：整体正例率 `0.6886`
- `hint_1 -> need_hint_2+`：整体正例率 `0.4985`
- `hint_2 -> need_hint_3+`：整体正例率 `0.0126`
- `hint_3 -> unsolved`：整体正例率 `0.0238`

这说明：

1. 最大的坍塌发生在一开始。
   - `base` 阶段就有约 `68.9%` 样本需要至少 1 层 hint。

2. 第二大的坍塌发生在进入 `hint_1` 之后。
   - 进入 `hint_1` 的样本里，大约一半还要继续掉到 `hint_2+`。

3. 一旦已经进入第三层 prefix，问题迅速从“普遍难”变成“局部病灶”。

因此如果你的目标是缓解“前几个 token 不 hint 就整条 rollout 崩掉”，最优先看的不是 depth 3 / 4，而是：

- root token 的第一跳路由
- depth-2 parent 下的局部竞争

## 3. 哪些节点学得差

### 3.1 root token：差的不是少数极端点，而是一整片尾部 root

在 [rollout_node_stats.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/rollout_node_stats.csv) 里，`base_to_need_hint` 的高风险 root token 包括：

- `<a_226>`: `197 / 217 = 90.78%`
- `<a_106>`: `792 / 902 = 87.80%`
- `<a_198>`: `2981 / 3405 = 87.55%`
- `<a_32>`: `1716 / 2018 = 85.03%`
- `<a_234>`: `1954 / 2301 = 84.92%`
- `<a_157>`: `3764 / 4622 = 81.44%`

相对更安全的 root token 包括：

- `<a_241>`: `4350 / 12117 = 35.90%`
- `<a_77>`: `49.42%`
- `<a_217>`: `57.96%`
- `<a_253>`: `58.58%`

这里有两个重要点：

1. root 失败并不只发生在最罕见的 root 上。
   - 例如 `<a_157>` 很常见，但仍然非常难。

2. root 失败也不是只集中在极少数 token 上。
   - 最危险的前 10 个 root token 只占全部样本的 `21.1%`
   - 但覆盖了 `25.7%` 的 `base` 失败
   - 说明它们确实更难，但问题仍然是广泛分布的，不是少数 root 的孤立事故

### 3.2 depth-2 node：高风险节点很多，但它们背后是少数高风险 parent

高风险 depth-2 node 示例：

- `(<a_125>, <b_167>)`: `54 / 55 = 98.18%`
- `(<a_198>, <b_23>)`: `46 / 48 = 95.83%`
- `(<a_157>, <b_184>)`: `41 / 43 = 95.35%`
- `(<a_241>, <b_154>)`: `40 / 42 = 95.24%`
- `(<a_253>, <b_254>)`: `72 / 76 = 94.74%`

但如果只看具体 node，会误以为困难很分散。实际上真正的结构问题在它们的 parent。

在 [rollout_parent_stats.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/rollout_parent_stats.csv) 中，`hint1_to_need_hint2_plus` 最糟的 parent 包括：

- `<a_253>`
  - `2713` 个样本
  - fail rate `74.60%`
  - sibling `105`
  - parent maxshare `0.0675`
- `<a_157>`
  - fail rate `73.09%`
  - sibling `99`
  - parent maxshare `0.0683`
- `<a_145>`
  - fail rate `66.19%`
  - sibling `83`
- `<a_194>`
  - fail count 最高：`3057`
  - fail rate `61.29%`
  - sibling `90`
- `<a_241>`
  - fail count `2519`
  - sibling `70`

所以 depth-2 的真实图景更像：

- 不是几万个零散坏 node
- 而是一批 overloaded parent，下面再挂着大量高风险 child

## 4. 这些坏节点跟频率、子树大小、兄弟数到底是什么关系

这部分是本次分析的核心。

### 4.1 root 层：频率和 task-conditioned dominance 越低，越容易 base 失败

在 root 级别，`fail_rate` 与特征的相关性大致是：

- `global_count_d1`
  - Pearson `-0.599`
- `task_parent_share_d1`
  - Pearson `-0.614`
- `subtree_d1`
  - Pearson `-0.337`
- `child_rank_d1`
  - Pearson `+0.371`

解释：

- root token 越常见，通常越容易。
- root token 在 task 条件下越 dominant，通常越容易。
- root token 排名越靠后，通常越难。
- 但 root 的 `subtree` 作用没有前两个强。

这说明 base 阶段最核心的不是“这个 root 后面有多少后代”，而是：

- 当前 task 条件下，这个 root token 是否足够像一个明显的第一跳选择

从 [rollout_feature_bin_summary.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/rollout_feature_bin_summary.csv) 看：

- 最低 `task_parent_share_d1` 五分位的 fail rate 是 `77.48%`
- 最高五分位降到 `52.93%`

而阈值分析显示：

- `task_parent_share_d1 <= 0.0341`
  - 覆盖 `43.4%` 的样本
  - 这些样本的 fail rate 是 `78.29%`
  - 覆盖了 `49.4%` 的全部 base 失败

也就是说：

- 近一半的 `base` 失败，都发生在“root token 在该 task 下不够 dominant”的那部分样本里

### 4.2 depth-2 parent：兄弟数几乎就是主导变量

在 depth-2 parent 级别，`fail_rate` 与结构特征的相关性非常强：

- `sibling`
  - Pearson `+0.934`
  - Spearman-like `+0.879`
- `parent_maxshare`
  - Pearson `-0.532`

这已经不是“有一点关系”，而是几乎接近结构主导。

解释成一句话就是：

- 第二层 parent 下面 child 越多，而且 top child 越不 dominant，`hint_1 -> need_hint_2+` 就越容易继续失败

这一点从五分位分箱也能看出来：

- `sibling_d2` 最低五分位：fail rate `28.23%`
- `sibling_d2` 最高五分位：fail rate `66.42%`

并且阈值 `sibling_d2 >= 62`：

- 覆盖 `74.8%` 的进入 `hint_1` 样本
- 这些样本 fail rate `56.0%`
- 覆盖了 `84.0%` 的全部 `hint_1 -> hint_2+` 失败

这组数的含义非常直接：

- 进入 `hint_1` 后的大部分失败，其实可以用一句话概括：
- “第二层 parent 太胖了”

### 4.3 depth-2 node：局部 task share 比全局频率更重要

在 depth-2 node 级别，`fail_rate` 与特征的相关性是：

- `task_parent_share_d2`
  - Pearson `-0.410`
  - Spearman-like `-0.536`
- `sibling_d2`
  - Pearson `+0.500`
- `subtree_d2`
  - Pearson `+0.258`
- `global_count_d2`
  - Pearson `-0.099`

这说明：

- 到第二层以后，单看 token 的全局频率已经不太够了。
- 更关键的是：
  - 它在当前 parent 下是否足够 dominant
  - 当前 parent 下面到底塞了多少兄弟

换句话说：

- root 层比较像“第一跳路由问题”
- 第二层更像“局部 codebook 竞争问题”

### 4.4 depth-3 以后：问题迅速从系统性变成局部病灶

`hint2_to_need_hint3_plus` 失败的集中度非常高：

- top 5 个 depth-3 parent 只占 `3.1%` 的样本
  - 却覆盖了 `50.6%` 的全部 `hint_2 -> hint_3+` 失败
- top 10 个 depth-3 parent 占 `6.1%` 样本
  - 覆盖 `77.5%` 失败
- top 20 个 depth-3 parent 占 `9.3%` 样本
  - 覆盖 `98.7%` 失败

代表性高风险 depth-3 parent：

- `<a_253><b_240>`
  - fail count `71`
  - fail rate `34.8%`
  - sibling `15`
- `<a_241><b_145>`
  - fail count `47`
  - fail rate `19.4%`
  - sibling `14`
- `<a_125><b_20>`
  - fail count `41`
  - fail rate `17.9%`
  - sibling `22`
- `<a_65><b_80>`
  - fail count `33`
  - fail rate `26.4%`

这说明一个逐层收缩的结构：

- `base` 失败：比较分散
- `hint_1` 失败：被少数 overloaded depth-2 parent 主导
- `hint_2` 失败：快速收缩到极少数 depth-3 prefix
- `hint_3` 未解：几乎变成单一病灶 branch

## 5. 为什么“前几个 token 不 hint 就一整个都 rollout 不出来”

这一部分是基于上述统计做的机制解释，我这里明确标注为推断。

### 5.1 第一层：第一跳不够 dominant，base 很容易直接走错大方向

证据：

- `base -> need_hint` 最有解释力的特征是 `task_parent_share_d1`
- sequence 特征明显更弱
- root fail rate 跟 `task_parent_share_d1` / `global_count_d1` 强负相关

推断：

- `base` 失败的很多样本，不是模型已经“知道大致分支，只是后面 rollout 坏了”
- 而是一开始第一跳就不够确信

一旦第一跳选错，后面 token 都是沿错误 prefix 自回归展开，恢复空间非常小。

### 5.2 第二层：即使第一跳选对，很多 parent 下面仍然太拥挤

证据：

- depth-2 parent 的 sibling 与 fail rate 相关性接近 `0.93`
- 很多高风险 parent 下面有 `70~105` 个 child
- 且 `parent_maxshare` 普遍只有 `0.05~0.11`

推断：

- 第一层 hint 的真正作用，不只是告诉模型一个 token
- 更像是把搜索空间从“整棵树”缩到了某个局部 parent

但如果这个 parent 本身就特别胖，模型仍然要在几十个近似平权 child 中竞争，所以第二层继续崩是很自然的。

### 5.3 前两个 token 之所以关键，是因为它们决定了后面是否进入一个可恢复区域

从结构上看：

- depth 3 以后大多数 parent 已经很稀疏
- 真正的巨大分叉主要发生在前两层

因此前两个 token 的作用不是“普通位置上的两个 token”，而是：

- 它们基本决定了你后面是在一个窄分支里 rollout
- 还是在一个巨大局部 codebook 里继续猜

所以“不给前几个 token hint 就 rollout 不出来”其实可以翻译成：

- 当前编码树把最难的决策过早地放在了 rollout 前半段

## 6. 这和 task 的关系

从节点级输出里也能看到，`hisTitle2sid` 和 `title_desc2sid` 在不少高风险 parent 上都会更差。

例如 depth-2 parent：

- `<a_253>`
  - `sid`: fail rate `69.88%`
  - `hisTitle2sid`: `92.17%`
  - `title_desc2sid`: `94.97%`
- `<a_194>`
  - `sid`: `56.96%`
  - `hisTitle2sid`: `81.47%`
  - `title_desc2sid`: `96.05%`
- `<a_241>`
  - `sid`: `56.11%`
  - `hisTitle2sid`: `57.76%`
  - `title_desc2sid`: `94.67%`

这说明：

1. 结构问题是共享的。
   - 这些 parent 本身就很难。

2. 输入通道差异会进一步放大结构问题。
   - 同一个坏 parent，对不同 task 的打击强度不同。

因此如果后续要做修复，不应该只看 task，也不应该只看 tree，而应该把两者一起考虑。

## 7. 这次分析支持哪些干预方向

下面不是“已经验证的 fix”，而是基于当前根因分析得出的优先级建议。

### 7.1 最值得优先试的是：对前两层做 branch-aware 的训练或推理干预

因为根因已经很清楚：

- 第一层是第一跳路由不稳
- 第二层是 overloaded parent 竞争太强

所以优先级最高的不是修最后 residual，而是改善前两层的决策。

可以考虑的方向：

1. 前两层单独加权
   - 对 depth-1 / depth-2 token loss 增权
   - 让模型更明确地把容量花在第一跳和第二跳

2. parent-aware 采样或 curriculum
   - 对高 sibling、低 parent_maxshare 的 parent 增加训练曝光
   - 不是单纯按 item 频率采样，而是按“危险 parent”采样

3. 推理时前两层做 branch-aware reranking
   - 不必对整条序列都改
   - 只在 depth 1 / 2 对候选 child 做额外 rerank 或 constrained expansion

### 7.2 如果允许改树，最该动的是第二层的负载均衡

当前最像结构瓶颈的是 depth 2：

- mean sibling `56.8`
- median sibling `62`
- max sibling `105`

这意味着树虽然理论深度只有 4，但第二层已经承担了过大的局部分类负担。

如果后续可以调整 codebook / tree construction，最有价值的问题是：

- 能不能让 depth-2 parent 更均衡
- 减少 `70~100+` child 的超胖 parent
- 尽量把高混淆 family 往后分散，而不是在第二层就拥挤地堆在一起

### 7.3 单纯补最后 residual，不会解决“前几个 token rollout 崩掉”

这一点很重要。

虽然最终 residual 很集中，但当前主痛点不是 residual。

真正吞掉大多数样本的是：

- `base -> need_hint`
- `hint_1 -> need_hint_2+`

所以如果后续资源有限，优先级应该是：

1. 先修第一跳路由
2. 再修第二层 parent overload
3. 最后再处理 residual branch pathology

## 8. 我认为最像根因的最终表述

如果要把这次探索压缩成一句工程判断，我会这样写：

- 当前 SID tree 的前两层没有形成“先粗分、后细分”的健康层次，而是把大量高歧义决策提前到了 rollout 前半段，导致模型在没有 hint 的情况下，第一跳容易不稳，第二跳又经常落入超胖 parent，于是整条路径很容易从开头就跑偏。

## 9. 怎么继续往下做

如果下一步继续做，我建议按这个顺序：

1. 针对 `base_to_need_hint`：
   - 抽取高风险 root token
   - 看这些 root 在训练时的 logits margin / beam 排名是否普遍不稳

2. 针对 `hint1_to_need_hint2_plus`：
   - 重点盯 `<a_253>`、`<a_157>`、`<a_145>`、`<a_194>`、`<a_241>` 这些超胖 parent
   - 看它们下面 child 的预测分布是否普遍过平

3. 如果允许改训练：
   - 先做前两层 loss reweight 或 branch-aware sampling

4. 如果允许改树：
   - 先评估 depth-2 parent 的重平衡收益

## 10. 复现命令

```bash
/Users/fanghaotian/Desktop/src/GenRec/.venv/bin/python \
  /Users/fanghaotian/Desktop/src/GenRec/scripts/hint_research/analyze_local_rollout_nodes.py
```
