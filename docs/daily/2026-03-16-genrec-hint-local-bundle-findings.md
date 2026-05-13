# GenRec Hint 本地 Bundle 研究结论（2026-03-16）

这份文档是对 `Instruments-grec` hint 研究 bundle 的一次本地复分析说明。原始 bundle 位于：

- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle`

这版中文文档的目标不是逐句翻译英文版，而是把下面几件事讲清楚：

1. 这个 bundle 里到底有什么数据。
2. `hint depth`、`effective_hint_depth`、`transition` 这些词分别是什么意思。
3. 新增的那些 CSV / JSON 分别在回答什么问题。
4. 这些数字最后支持了怎样的结论。

如果你只想先抓住一句话版本，可以先记住下面这段：

- `hint depth` 不是单一的“难度分数”，它混合了两类东西：任务输入本身的歧义，以及 SID 树局部分支竞争的难度。
- `hisTitle2sid` 的确比 `sid` 难，而且这种更难并不能只用局部树结构解释掉。
- 一旦样本已经需要更深的 hint，后面的主要矛盾就不再是“序列任务本身难不难”，而是“当前 prefix 下 sibling 太多、当前 child 不够 dominant”。
- 最后那 `11` 个完全没解开的样本并不是散落在很多地方，而是几乎全挤在同一条 branch 上，并且又被其中一个 leaf `<d_100>` 主导。

## 1. 研究对象到底是什么

这次复分析用到的是 bundle 内已经打包好的静态结果，不是重新跑训练或重新生成：

- `train.json`
- `id2sid.json`
- `summary/details.json`
- `instruments_grec_beam16_hint_difficulty_table.csv`
- `Instruments.item.json`
- `Instruments.inter.json`

这轮复分析新增的脚本是：

- [explore_local_hint_bundle.py](/Users/fanghaotian/Desktop/src/GenRec/scripts/hint_research/explore_local_hint_bundle.py)

这个脚本读取 bundle 里的原始难度表，再补上按 SID 深度对齐的局部结构特征，然后导出新的分析结果到：

- `/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis`

## 2. 先把最容易混淆的术语讲清楚

### 2.1 `effective_hint_depth` 是什么

这是整份分析里最重要的字段，可以把它理解成：

- `0`：`base` 阶段就命中了，不需要任何 hint。
- `1`：`base` 没解出来，但给了第 1 层 hint 后命中。
- `2`：需要看到第 2 层 hint 才命中。
- `3`：需要看到第 3 层 hint 才命中。
- `4`：跑完 `hint_3` 仍然没命中，也就是最终未解。

在这份 bundle 里，总样本数是 `106,441`。按 `effective_hint_depth` 分布：

- `0`: `33,147`
- `1`: `36,756`
- `2`: `36,076`
- `3`: `451`
- `4`: `11`

这说明一件很重要的事：

- 绝大多数样本的问题会在 `base / hint_1 / hint_2` 这三个层次里解决。
- 真正走到 `hint_3` 甚至最后仍未解的样本，占比极小，所以这些样本已经值得单独按 branch pathology 来看，而不是继续当作“普通 hard case”。

### 2.2 `transition` 是什么

这次复分析不再把所有 hint 样本混在一起，而是拆成四个阶段问题：

- `base -> need_hint`
  - 问的是：一个样本在 `base` 阶段会不会失败，从而需要至少 1 层 hint。
- `hint_1 -> need_hint_2+`
  - 问的是：一个已经进入 `hint_1` 的样本，是否还要继续要到第 2 层 hint。
- `hint_2 -> need_hint_3+`
  - 问的是：一个已经进入 `hint_2` 的样本，是否还要继续要到第 3 层 hint。
- `hint_3 -> unsolved`
  - 问的是：一个已经走到 `hint_3` 的样本，最后会不会依然失败。

为什么要这样拆？

- 因为每个阶段真正决定成败的 token 深度不一样。
- `base -> need_hint` 真正对应的是第 1 层 token。
- `hint_1 -> need_hint_2+` 真正对应的是第 2 层 token。
- `hint_2 -> need_hint_3+` 真正对应的是第 3 层 token。
- `hint_3 -> unsolved` 真正对应的是第 4 层 token。

只有这样按 stage 对齐，才不会把不同深度的“决定性 token”混成一锅。

### 2.3 这几个结构特征分别在说什么

下面这些字段会在文档和 CSV 里反复出现：

- `task_parent_share_dk`
  - 在某个 task 条件下，当前 parent 下这个 child 的占比。
  - 可以把它理解成“在这个任务语境里，这个 token 在本地 prefix 下是不是主流选项”。

- `child_share_dk`
  - 不分 task，只看整棵树里当前 parent 下这个 child 的占比。

- `subtree_dk`
  - 当前 token 对应的子树规模。

- `sibling_dk`
  - 当前 parent 下有多少个兄弟节点。
  - 数越大，局部竞争通常越激烈。

- `parent_maxshare_dk`
  - 当前 parent 下最强那个 child 的占比。
  - 值越高，说明这个 parent 下存在明显 dominant child；值越低，说明大家都差不多。

- `parent_entropy_dk`
  - 当前 parent 下子节点分布是否均匀。
  - 越高通常说明越平均、越难靠 dominance 区分。

- `child_rank_dk`
  - 当前 child 在 parent 内按频次排第几。
  - 排名越靠后，通常越不 dominant。

- `lift`
  - 某个 group 的正例率，相对于该 transition 全局平均正例率的倍率。
  - 例如 `lift = 2`，可以粗略理解为“这个 group 的风险约是整体平均的两倍”。

- `rate_gap_his_minus_sid`
  - 同一个 group 里，`hisTitle2sid` 的风险率减去 `sid` 的风险率。
  - 正值越大，说明这个 group 对 `hisTitle2sid` 额外更伤。

- `balanced_acc`
  - 一个单特征阈值把正负样本分开的能力。
  - `0.5` 基本等于没什么区分力，接近 `1.0` 才说明几乎能单独把这一步解释掉。

## 3. 这次到底修正了原 notebook 的什么问题

在真正解释数字前，先说这次复分析为什么有必要。

原 notebook 里有两个容易误导结论的地方：

### 3.1 `required_*` 特征把不同决策深度混在了一起

例如：

- 一个 `effective_hint_depth = 1` 的样本，真正关键的是第 1 层 token。
- 一个 `effective_hint_depth = 2` 的样本，真正关键的是第 2 层 token。

如果直接把这两类样本一起比较，就会出现一种假象：

- 好像“deep hint 样本”在某些 token 统计上有统一特征。

但很多时候，这其实只是因为你拿第 1 层 token 和第 2 层 token 在同一张表里做了比较。

### 3.2 原来的 `deep-hint token lift` 表其实口径不对齐

在旧表里：

- 很多 `hint_1` 行贡献的是 `<a_*>`
- 很多更深 hint 行贡献的是 `<b_*>` 或 `<c_*>`

因此那种 `share_lift = 1.0` 的结果，并不能直接解释成“没有 concentration”，很多时候只是：

- 你比较的是不同层级的 token
- 所以分组本身不对齐

### 3.3 这次改成的口径

这次统一改成按 transition 对齐的分析方式：

- `base -> need_hint` 只看 depth-1 特征
- `hint_1 -> need_hint_2+` 只看 depth-2 特征
- `hint_2 -> need_hint_3+` 只看 depth-3 特征
- `hint_3 -> unsolved` 只看 depth-4 特征

这一步非常关键。因为现在每一步都严格对应“下一次 hint 会暴露哪一层 token”，所以你可以把每个 transition 视为一个局部决策问题，而不是一个混合口径的总难度分数。

## 4. 先看整体难度分布

### 4.1 三个 task 的总体差异

从 [hint_transition_summary.json](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_summary.json) 和原始难度表可以得到：

| task | 样本数 | base 命中率 | 需要至少 1 层 hint | 需要至少 2 层 hint | 需要至少 3 层 hint | 最终未解率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sid` | 84,890 | 28.25% | 71.75% | 32.70% | 0.35% | 0.0071% |
| `hisTitle2sid` | 10,000 | 8.69% | 91.31% | 61.24% | 0.78% | 0.0100% |
| `title_desc2sid` | 11,551 | 71.81% | 28.19% | 22.95% | 0.78% | 0.0346% |

直观解释：

- `hisTitle2sid` 最难。
  - 它几乎一开始就大面积需要 hint。
- `sid` 中等。
  - 有不少样本 base 能解，但也有大量样本会掉进 hint_1 / hint_2。
- `title_desc2sid` 在 base 上最容易。
  - 但这不代表它整体简单，因为一旦 base 没解出来，后面继续恶化得很快。

### 4.2 continuation rate 更能看出 task 的“形状”

如果只看进入某阶段后的续跌概率：

- `sid`
  - 进入 `hint_1` 后，继续需要 `hint_2+` 的比例：`45.58%`
  - 进入 `hint_2` 后，继续需要 `hint_3+` 的比例：`1.06%`
  - 进入 `hint_3` 后，最终未解比例：`2.04%`

- `hisTitle2sid`
  - 进入 `hint_1` 后，继续需要 `hint_2+` 的比例：`67.07%`
  - 进入 `hint_2` 后，继续需要 `hint_3+` 的比例：`1.27%`
  - 进入 `hint_3` 后，最终未解比例：`1.28%`

- `title_desc2sid`
  - 进入 `hint_1` 后，继续需要 `hint_2+` 的比例：`81.42%`
  - 进入 `hint_2` 后，继续需要 `hint_3+` 的比例：`3.39%`
  - 进入 `hint_3` 后，最终未解比例：`4.44%`

这组数很有意思：

- `title_desc2sid` 虽然在 `base` 上最容易，但一旦没有在 `base` 解决，后续 continuation rate 反而最高。
- 所以它更像“双峰分布”：
  - 容易的样本非常容易
  - 难的样本会迅速滑入局部病态 branch

## 5. 假设 1：任务难度不能被局部 tree 结构完全解释

结论：支持。

这一节最容易让人困惑的地方，是“标准化后”到底是什么意思。先给一句最直白的话：

- 这一节想回答的不是“`hisTitle2sid` 难不难”，而是“它是不是只是因为自己更常掉进坏 branch，所以看起来更难”。

如果答案是“是”，那么只要把 `sid` 和 `hisTitle2sid` 放到相似的局部结构条件里比较，差距就应该明显缩小甚至消失。

### 5.1 先看原始差距

原始 task gap 是：

- `sid`: `need_hint_rate = 71.75%`，`hint2_plus_rate = 32.70%`
- `hisTitle2sid`: `need_hint_rate = 91.31%`，`hint2_plus_rate = 61.24%`

只看这组数，我们只能说：

- `hisTitle2sid` 比 `sid` 更容易失败

但这里还不能判断，这种更难到底来自：

1. `hisTitle2sid` 更常落到坏 branch
2. 还是即使落在同类 branch 上，它自己也更难

### 5.2 这里的“标准化后”到底怎么算

这次脚本里做的事，其实不是复杂模型，而是一个很朴素的“同条件重加权比较”。

以某个 transition 为例，比如 `base -> need_hint`：

1. 只取这一阶段的样本。
2. 用和这一阶段对应的局部结构特征给样本分箱。
   - 这里实际用了两个维度：
   - `task_parent_share_d1`
   - `subtree_d1`
3. 每个维度按分位数切成若干档，再组合成一个个小格子，也就是文里说的 `bin`。
4. 只保留 `sid` 和 `hisTitle2sid` 在这个格子里都出现过样本的格子。
5. 在每个共同格子里，分别计算：
   - 这个格子里 `sid` 的失败率
   - 这个格子里 `hisTitle2sid` 的失败率
6. 最后用相同的一套格子权重，对两个 task 分别做加权平均。

可以把它想成下面这个意思：

- 先不让一个 task 靠“自己样本分布碰巧更差”占便宜或吃亏
- 而是尽量把两边塞进相似的局部结构环境里
- 再比较在这些环境里，谁还是更容易失败

### 5.3 一个不带公式的玩具例子

假设只有两种局部结构格子：

- `A`：容易的 branch
- `B`：困难的 branch

并且：

- `sid` 大多数样本在 `A`
- `hisTitle2sid` 大多数样本在 `B`

那么原始总体失败率里，确实会混入一个“样本分布不同”的影响：

- 不是它本身更难
- 而是它更常掉到坏地方

所谓“标准化后”，就是强行让两个 task 都按同样的 `A/B` 比例重新算一次平均失败率。

如果这样重算后 gap 消失了，说明：

- 原来的差距主要是 branch 组成差异造成的

如果这样重算后 gap 还在，说明：

- 即使放在同类 branch 里，`hisTitle2sid` 还是更难

### 5.4 真正算出来的结果

如果按照 stage-aligned 的局部结构 bin 去匹配 `sid` 和 `hisTitle2sid`，结果是：

- `base -> need_hint`
  - 原始：`sid 0.717`，`hisTitle2sid 0.913`
  - 标准化后：`sid 0.788`，`hisTitle2sid 0.957`
- `hint_1 -> need_hint_2+`
  - 原始：`sid 0.456`，`hisTitle2sid 0.671`
  - 标准化后：`sid 0.460`，`hisTitle2sid 0.711`
- `hint_2 -> need_hint_3+`
  - 原始：`sid 0.0106`，`hisTitle2sid 0.0127`
  - 标准化后：`sid 0.0223`，`hisTitle2sid 0.0287`

这里再强调一次，这个“标准化后”不是：

- 一个新的真实命中率
- 也不是“完全控制了所有因素后的纯净因果效应”

它更准确地说是：

- 只在双方共同出现过的局部结构格子里
- 用同样的格子权重
- 重算出来的对比率

所以你会看到一个看起来反直觉的现象：

- 两边的“标准化后”数值有时会都比原始值更高

这并不矛盾。它通常只是说明：

- 双方共同覆盖到的那些格子，本来就比各自全体样本更难

真正要看的不是“标准化后为什么变高了”，而是：

- 在相同格子分布下，两者的 gap 有没有消失

### 5.5 这组数字真正说明了什么

把 gap 单独拿出来看会更清楚：

- `base -> need_hint`
  - 原始 gap：`0.913 - 0.717 = 0.196`
  - 标准化后 gap：`0.957 - 0.788 = 0.169`
- `hint_1 -> need_hint_2+`
  - 原始 gap：`0.671 - 0.456 = 0.215`
  - 标准化后 gap：`0.711 - 0.460 = 0.251`
- `hint_2 -> need_hint_3+`
  - 原始 gap：`0.0127 - 0.0106 = 0.0021`
  - 标准化后 gap：`0.0287 - 0.0223 = 0.0064`

所以这节真正支持的结论是：

- 局部 branch 结构当然重要，因为一做这种重加权，绝对数值会发生变化。
- 但 gap 并没有被“解释掉”。
- 在 `base -> need_hint` 这一步，gap 变小了一点，但仍然很大。
- 在后两步里，gap 甚至没有缩小，反而在共同结构格子里更明显。

结果说明：

- 局部 branch 结构确实重要
- 但它解释不掉全部 task gap
- 更准确地说，是“按 `task_parent_share + subtree` 这类局部结构做粗粒度重加权后，task gap 仍然存在”
- 所以 `hisTitle2sid` 的困难不只是样本更常落到坏 branch 这么简单

换成更直白的话就是：

- `hisTitle2sid` 不是单纯因为“恰好落到了更坏的 SID 分支”才更难
- 更像是：哪怕把它和 `sid` 放到差不多形状的局部树环境里，它还是更容易继续要 hint
- 因此，任务输入形式本身大概率额外引入了歧义

### 5.6 这一节不要过度解读什么

为了避免误读，这里还要补一句边界条件。

这一节并不是在证明：

- “局部 tree 结构不重要”
- 或者“task 输入一定是唯一原因”

它真正证明的是一个更窄的命题：

- 仅用这次纳入标准化的局部结构特征，没法把 `sid` 和 `hisTitle2sid` 的难度差解释干净

也就是说，剩下来的 gap 可能来自：

- 任务输入通道差异
- 还没被纳入的其他结构特征
- 或者两者共同作用

## 6. 假设 2：更深 hint 在不同 stage 上代表的不是同一种“难”

结论：支持。

这一部分最重要，因为它回答了：

- `hint_2` 和 `hint_3` 到底是不是同一种难度的延长？

答案是：不是。

### 6.1 `base -> need_hint`：更像第一个 token 的路由问题

最强单特征是：

- `task_parent_share_d1`，balanced accuracy `0.601`

其次是：

- `global_count_d1`，balanced accuracy `0.591`
- `subtree_d1`，balanced accuracy `0.575`

而 sequence 特征在这里很弱：

- 最好的 sequence 特征 balanced accuracy 也只有约 `0.526`

怎么理解这组数？

- 在最初这一步，模型失败并不主要是因为“序列位置太晚”或“history 太长”。
- 更像是它一开始就没把 root child 路由到足够 dominant 的方向。
- 也就是说，第一个 token 是不是本地强势候选，比 sequence 位置更重要。

注意 `0.601` 这个 balanced accuracy 并不算非常高，这意味着：

- 第 1 层局部结构能解释一部分 base 失败
- 但远远没有到“一眼就能分开”的程度

### 6.2 `hint_1 -> need_hint_2+`：开始进入局部 codebook 竞争

最强单特征：

- `task_parent_share_d2`，balanced accuracy `0.620`
- `sibling_d2`，balanced accuracy `0.592`
- `parent_entropy_d2`，balanced accuracy `0.590`

这一步的直观含义是：

- 第一个 token 已经被 hint 揭示
- 现在真正的问题变成：在这个已知 prefix 下，第 2 层 token 有没有足够明显的本地优势

所以这一步已经不像“泛化难度”了，更像：

- 当前局部 codebook 里，竞争是否太激烈

### 6.3 `hint_2 -> need_hint_3+`：几乎是纯局部分支竞争

最强单特征：

- `sibling_d3`，balanced accuracy `0.838`
- `task_parent_share_d3`，balanced accuracy `0.761`
- `child_share_d3`，balanced accuracy `0.748`
- `subtree_d3`，balanced accuracy `0.726`

这里和前两步最大的区别是：

- 结构信号已经强得多了

尤其是：

- `sibling_d3 >= 13` 就能给出很强的分离效果
- 说明到了第 3 层 prefix，问题已经高度集中在“同一 parent 下兄弟太多”这种局部现象上

换句话说：

- 一旦样本已经掉到 `hint_2`
- 后面是否还要 `hint_3`
- 基本主要看这一小块 prefix 下的竞争有多拥挤，而不是再看 task 的宏观性质

### 6.4 `hint_3 -> unsolved`：最终 residual 几乎就是叶子级歧义

最强单特征：

- `child_share_d4 <= 0.0476`，balanced accuracy `0.987`
- `sibling_d4 >= 21`，balanced accuracy `0.987`
- `parent_maxshare_d4 <= 0.0476`，balanced accuracy `0.987`

这组数几乎已经是在说：

- 最终没解开的样本，几乎都位于一个 leaf 数很多、而且每个 leaf 占比都很接近的 parent 下

这不再是“模型整体还不够强”的故事，而是：

- 最后一层叶子选择存在高度集中的局部歧义

## 7. 假设 3：最终 hard case 是高度集中的，不是漫散的

结论：强烈支持。

从 [hint_transition_residual_cases.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_residual_cases.csv) 可以直接看到：

- 最终 residual 一共只有 `11` 个样本
- task 分布是：
  - `sid`: `6`
  - `title_desc2sid`: `4`
  - `hisTitle2sid`: `1`

更关键的是：

- 这 `11` 个样本全部都落在同一个 depth-4 parent prefix：
  - `<a_65><b_80><c_183>`

这个 parent 下总共有 `21` 个 leaf item。

所以 final residual 不是：

- 很多地方各坏一点

而是：

- 一条 branch 特别坏

## 8. `dominant residual branch` 的内部结构是什么

### 8.1 它不只是一个坏 parent，还是一个被坏 leaf 主导的 parent

从 [dominant_residual_parent_leaf_stats.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_leaf_stats.csv) 可以看到：

- `<d_100>`
  - `total_case_count = 6`
  - `unsolved_case_count = 6`
  - leaf 内部未解率 = `1.0`

其它 residual leaf 远没有这么夸张：

- `<d_229>`: `1 / 4`
- `<d_239>`: `1 / 4`
- `<d_1>`: `1 / 7`
- `<d_137>`: `1 / 7`
- `<d_88>`: `1 / 9`

所以这条坏 branch 还可以继续拆成两层：

1. 一个 parent prefix 本身就很拥挤。
2. 其中又有一个 leaf `<d_100>` 对失败贡献最大。

### 8.2 这个 parent 实际上是一整个非常紧的商品家族

[dominant_residual_parent_items.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_items.csv) 展示了这 `21` 个 leaf 的真实商品标题。

它们基本都属于：

- `Walker & Williams` 吉他背带家族

例如：

- `Walker & Williams G-26 Black Semi-Gloss Bullnose Guitar Strap...` 对应 `<d_100>`
- `Walker & Williams G-44 Cabernet Red Guitar Strap...` 对应 `<d_1>`
- `Walker & Williams G-43 Cognac Brown Guitar Strap...` 对应 `<d_239>`

也就是说，这不是一个随机杂糅的 parent，而是标题和语义都非常接近的一整簇商品。

### 8.3 更关键的反证：`<d_100>` 不是最稀有，也不是词面最模糊

这是这轮分析里最值得注意的点之一。

从 [dominant_residual_parent_leaf_distinctiveness.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_leaf_distinctiveness.csv) 可以看到，灾难 leaf `<d_100>`：

- 不是词面最空泛的 leaf
- 反而有比较明显的独有词
  - `bullnose`
  - `gloss`
  - `semi`
- `unique_term_count = 3`
- `mean_jaccard_to_siblings = 0.0`

它也不是最稀有的 leaf：

- `<d_100>` 的 `global_count_d4 = 360`
- 但一些已经能解开的 leaf 更低频：
  - `<d_59>`: `240`
  - `<d_88>`: `253`
  - `<d_192>`: `211`

所以两个最容易想到的解释都不够：

- “它失败是因为标题太模糊”
- “它失败是因为 leaf 太稀有”

更合理的解释更接近：

- 这一家商品在表示空间或索引空间里先被压进了一个过紧的 branch
- 然后这个 branch 内部又对 `<d_100>` 形成了稳定偏置
- 但这个 leaf 自己并不具备最差的表面文本特征

## 9. 在最终 residual 之前，困难样本也已经表现出 branch concentration

这说明最后那条病态 branch 不是突然出现的，它前面已经有征兆。

### 9.1 `hint_1 -> need_hint_2+` 的高风险 group

从 [hint_transition_top_groups.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_top_groups.csv) 看，下面这些 `(parent_d2, token_d2)` group 的风险已经非常高：

- `(<a_157>, <b_8>)`
  - `39 / 39` 都继续需要 `hint_2+`
- `(<a_157>, <b_249>)`
  - `32 / 32`
- `(<a_198>, <b_153>)`
  - `30 / 30`
- `(<a_157>, <b_40>)`
  - `27 / 27`
- `(<a_241>, <b_56>)`
  - `25 / 25`

这意味着：

- 一旦 prefix 已经落到某些 depth-2 group，继续下沉几乎是必然事件

### 9.2 `hint_2 -> need_hint_3+` 的高风险 group

在 depth-3 层，集中性更强：

- `(<a_253><b_240>, <c_59>)`
  - `19 / 28` 继续需要 `hint_3+`
  - `lift = 53.67`
- `(<a_253><b_240>, <c_49>)`
  - `15 / 23`
  - `lift = 51.58`
- `(<a_241><b_145>, <c_84>)`
  - `18 / 32`
  - `lift = 44.49`
- `(<a_125><b_20>, <c_222>)`
  - `16 / 30`
  - `lift = 42.18`
- `(<a_65><b_80>, <c_210>)`
  - `9 / 26`
  - `lift = 27.38`
- `(<a_65><b_80>, <c_183>)`
  - `23 / 92`
  - `lift = 19.77`

特别注意最后一条：

- `(<a_65><b_80>, <c_183>)`

它正是最终残留 parent `<a_65><b_80><c_183>` 的上一级征兆。也就是说：

- 这条最终病态 branch 在 `hint_2` 阶段就已经开始显著冒头

## 10. 共享 hotspot 和 task 特异 gap 要分开看

这是这轮分析相比原 notebook 最有用的新拆分之一。

之前容易混在一起的，其实是两个不同问题：

1. 哪些 branch 对 `sid` 和 `hisTitle2sid` 都难。
2. 哪些 branch 对 `hisTitle2sid` 尤其更难。

### 10.1 共享热点 branch：说明两种任务会撞上同一批硬骨头

[shared_transition_hotspots.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/shared_transition_hotspots.csv) 给出的不是“谁更差”，而是“哪些地方两边都差”。

在 `base -> need_hint` 上，共享高风险 root token 包括：

- `<a_105>`
- `<a_32>`
- `<a_173>`
- `<a_19>`
- `<a_198>`
- `<a_65>`
- `<a_234>`
- `<a_157>`

在 `hint_1 -> need_hint_2+` 上，最强共享热点包括：

- `(<a_253>, <b_240>)`
- `(<a_241>, <b_197>)`
- `(<a_157>, <b_153>)`
- `(<a_194>, <b_247>)`
- `(<a_194>, <b_58>)`
- `(<a_65>, <b_2>)`

例如：

- `(<a_253>, <b_240>)`
  - `sid`: `177` 个样本，继续下沉率 `80.79%`
  - `hisTitle2sid`: `39` 个样本，继续下沉率 `97.44%`

在 `hint_2 -> need_hint_3+` 上，最强共享热点是：

- `(<a_241><b_197>, <c_210>)`
  - `sid` 风险率 `13.16%`
  - `hisTitle2sid` 风险率 `20.00%`
  - 相对各自基线的 lift 分别约 `12.43` 和 `15.70`

这说明：

- 两个 task 不是在完全不同的 SID 区域各自失败
- 它们会反复撞上同一批 shared hard-tree backbone

### 10.2 task-specific gap group：说明 `hisTitle2sid` 在某些 branch 上额外被放大

[transition_task_gap_groups.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/transition_task_gap_groups.csv) 回答的是：

- 在同一个局部 group 里，`hisTitle2sid` 比 `sid` 额外差多少

对 `base -> need_hint`，最大 gap 出现在 root：

- `<a_77>`: `0.8546 - 0.4607 = +0.3939`
- `<a_241>`: `+0.3208`
- `<a_59>`: `+0.3122`
- `<a_253>`: `+0.2920`
- `<a_125>`: `+0.2917`

对 `hint_1 -> need_hint_2+`，gap 更尖锐、也更局部：

- `(<a_65>, <b_194>)`
  - `sid`: `5.00%`
  - `hisTitle2sid`: `67.86%`
  - gap `+0.6286`
- `(<a_125>, <b_212>)`
  - gap `+0.5894`
- `(<a_192>, <b_246>)`
  - gap `+0.5669`
- `(<a_77>, <b_254>)`
  - gap `+0.5480`
- `(<a_241>, <b_170>)`
  - gap `+0.5361`

因此更准确的说法不是：

- `hisTitle2sid` 在所有地方都均匀更难

而是：

- 一部分 branch 本来就是 shared hotspot
- 其中又有一部分 branch 会对 `hisTitle2sid` 发生额外 amplification

## 11. 其它类似病灶 branch：还没死透，但结构已经危险

[parent_leaf_pathology_summary.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/parent_leaf_pathology_summary.csv) 和 [repeated_pathology_candidates.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/repeated_pathology_candidates.csv) 的作用是：

- 不只看已经 residual 的 branch
- 还去找“结构上和它很像，但还没坏到最后一步”的候选 branch

比较典型的候选有：

- `<a_253><b_240><c_49>`
  - 只有 `2` 个 leaf
  - 商品标题是 `Evans G1 Clear Drum Head, 18 Inch` 和 `13 Inch`
  - `15` 个 `hint3+` case 全都集中在同一个 leaf 上
  - 标题几乎完全一样，只差尺寸

- `<a_194><b_58><c_78>`
  - `5` 个 leaf
  - 都是 `GLS Audio` 的 `XLR male -> XLR female` patch cable 系列
  - family 内共享词非常多

- `<a_194><b_58><c_248>`
  - `7` 个 leaf
  - 也是 `GLS Audio` 的 microphone cable / patch cord 系列
  - 同样有极高的 family 内相似性

这些例子说明：

- 不一定只有最终 residual branch 才值得关注
- 还有一些 branch 已经表现出“family 很紧 + 晚期难度集中到单个 leaf”的结构
- 它们目前没形成 residual，只是因为还没坏到最后一步

## 12. 这份 output 目录里每个主要文件分别该怎么看

如果你想重新看证据，建议按下面顺序：

### 第 1 层：先看原始难度分布

- [instruments_grec_beam16_hint_difficulty_table.csv](/Users/fanghaotian/Desktop/src/GenRec/output/jupyter-notebook/genrec-hint-cascade-artifacts/instruments_grec_beam16_hint_difficulty_table.csv)
  - 最基础的样本级总表。
  - 如果你想知道“某个样本属于哪个 task、最早在哪一层命中、是否最终未解”，先看这个。

### 第 2 层：看 bundle 复分析给出的总摘要

- [hint_transition_summary.json](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_summary.json)
  - 这是最适合先看的总摘要。
  - 包含 task-level rate、标准化 task gap、residual parent 汇总。

### 第 3 层：看每个 stage 是由什么结构特征驱动的

- [hint_transition_feature_thresholds.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_feature_thresholds.csv)
  - 回答“哪一个单特征最能解释这一步”。
  - 适合用来判断每个阶段更像是频率问题、dominance 问题还是 sibling 竞争问题。

### 第 4 层：看哪些 group 风险最高

- [hint_transition_top_groups.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_top_groups.csv)
  - 回答“哪些 token / prefix group 的风险最高”。

- [shared_transition_hotspots.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/shared_transition_hotspots.csv)
  - 回答“哪些热点是 `sid` 和 `hisTitle2sid` 共同的”。

- [transition_task_gap_groups.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/transition_task_gap_groups.csv)
  - 回答“哪些 group 对 `hisTitle2sid` 的伤害被额外放大”。

### 第 5 层：看最终 residual 到底是谁

- [hint_transition_residual_cases.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/hint_transition_residual_cases.csv)
  - 直接列出最后 `11` 个没解开的样本。

- [dominant_residual_parent_items.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_items.csv)
  - 把那个坏 parent 下面的所有真实商品标题列出来。

- [dominant_residual_parent_leaf_stats.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_leaf_stats.csv)
  - 看哪个 leaf 真正主导失败。

- [dominant_residual_parent_leaf_distinctiveness.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/dominant_residual_parent_leaf_distinctiveness.csv)
  - 检查失败叶子是不是“最模糊”或“最稀有”。

### 第 6 层：找未来可能恶化的 branch

- [parent_leaf_pathology_summary.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/parent_leaf_pathology_summary.csv)
  - 对所有 parent 做结构病灶汇总。

- [repeated_pathology_candidates.csv](/Users/fanghaotian/Desktop/src/GenRec/output/local-research-bundles/instruments_grec_hint_research_bundle/analysis/repeated_pathology_candidates.csv)
  - 找和 residual branch 相似、但还没彻底坏掉的候选。

## 13. 最终应该怎么理解这组结果

把所有输出连在一起，可以得到一个比较稳定的故事：

1. `hint depth` 不是单一难度标量。
   它混合了至少两层机制：
   - task / 输入形式歧义
   - SID tree 的局部分支竞争

2. `hisTitle2sid` 的确更难。
   而且这种更难在控制局部树结构后仍然存在，所以不能简单归因于“它碰巧落在了更坏的 branch 上”。

3. 一旦样本已经需要 `hint_2`，后面的故事主要是结构性的。
   大 sibling 数、低 local share、弱 parent dominance，是更关键的信号。

4. 最终 residual 不是分散噪声，而是单点病灶。
   它先被一个 parent prefix `<a_65><b_80><c_183>` 主导，再被其中一个 leaf `<d_100>` 主导。

5. `<d_100>` 不是最稀有，也不是词面最模糊。
   所以问题更接近表示空间或索引空间的局部塌缩，而不只是表面文本特征不够强。

6. `title_desc2sid` 不能简单叫“容易”。
   它更像：
   - 容易样本很多，所以 base 看起来很好
   - 但一旦 base 失败，后面很容易掉进更深、更局部的病态 branch

## 14. 如果还要继续追，最值得做什么

按优先级排序，后续最值得追的方向是：

1. 继续追 `<d_100>` 为什么会在 `<a_65><b_80><c_183>` 中成为唯一灾难叶子。
2. 在其它高风险 candidate parent 上检查是否会复现“dominant bad leaf”模式。
3. 如果后续能拿到更底层的索引或表示信息，检查：
   - embedding / index 空间是否把这些 family 压得过近
   - hotspot branch 上 beam / logits 是否过早塌缩

## 15. 复现状态

这轮分析是探索性的，但当前是可复现的。脚本如下：

```bash
/Users/fanghaotian/Desktop/src/GenRec/.venv/bin/python \
  /Users/fanghaotian/Desktop/src/GenRec/scripts/hint_research/explore_local_hint_bundle.py
```

如果你现在要继续阅读这套结果，我建议实际顺序是：

1. 先看这份文档第 2、4、13 节，建立基本心智模型。
2. 再开 `hint_transition_summary.json` 看 task-level 数字。
3. 接着看 `hint_transition_feature_thresholds.csv` 和 `shared_transition_hotspots.csv`。
4. 最后只盯住 `dominant_residual_parent_*` 那几张表，把最终病灶 branch 看透。
