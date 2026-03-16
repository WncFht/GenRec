# GenRec 固定 Hint `rule_only` 训练状态记录（2026-03-15）

本文档用于记录当前 `GenRec` 在 **固定 oracle hint depth + `rule_only` RL** 方向上的背景、已有结论、工程实现、失败原因与当前判断。

这是一份**状态记录文档**，不是最终方案文档。因此它重点回答：

1. 我们当前到底在试什么。
2. 已经有哪些离线证据支持这个方向。
3. 代码已经改到了什么程度。
4. 当前为什么训练会挂住。
5. 各种实现路线分别意味着什么。

本文档**不包含“下一步建议”章节**，只记录到当前状态为止。

---

## 1. 当前研究目标

当前想验证的核心问题是：

> 如果我们已经离线知道“每个样本至少需要多少 hint depth，才能在 `beam=16` 下拿到至少一个 `rule hit`”，那么能否在 RL 训练中直接把这个 depth 当成该样本的固定 hint 深度，从而缓解 `rule_only` reward 的极端稀疏问题？

更具体地说：

- 当前训练仍然使用：
  - `beam search = 16`
  - `reward_mode = rule_only`
  - `num_train_epochs = 2`
- 当前想做的不是在线 adaptive hint scheduler，而是：
  - **固定 per-sample oracle hint depth**
- 这个 oracle depth 来自离线 analyzer：
  - `GenRec/analyze_rl_beam_hint.py`

因此，这个实验本质上是：

> **Fixed Oracle Hint-Depth Rule-Only RL**

它要回答的问题是：

- 给每个样本配上一个离线测得的“最小有效 hint 深度”，`rule_only` RL 是否更容易训练？

它**不直接回答**的问题是：

- 模型是否因此学会了无 hint 条件下自己生成这些 prefix
- fixed hint 本身是否是最终合理训练目标

---

## 2. 为什么会想到固定 Hint Depth

这个方向来自我们已经做完的离线 cascade 分析：

- `base`
- `hint_1`
- `hint_2`
- `hint_3`

在 `beam=16` 下，我们已经有了每个样本从 `base -> hint_1 -> hint_2 -> hint_3` 的 exact 命中情况。

这意味着对每个样本，我们其实已经能定义：

- `d=0`：无 hint 就能命中
- `d=1`：要 hint 1 个 token 才首次命中
- `d=2`：要 hint 2 个 token 才首次命中
- `d=3`：要 hint 3 个 token 才首次命中
- `unsolved`：hint 3 个 token 仍然不命中

也就是说，离线分析已经给出了一个 sample-wise 的“最小有效 scaffold 深度”。

于是自然想到：

> 既然 online RL 的 hardest case 主要是没梯度，那不如直接把这个离线测得的最小有效深度拿来作为训练时的真实 hint 深度。

---

## 3. 已有离线分析结论（对这个方向的支持）

离线 notebook 和 `summary.json` 已经给出几个非常强的结论。

### 3.1 `beam=16` 下，base frontier 仍然明显不够

`beam=16` 的 `base` exact-hit 率大约是：

- `31.14%`

说明即使用 `beam=16`，在 `rule_only` 下仍有大量样本整组 beam 拿不到正 reward。

### 3.2 `hint_1` 有帮助，但不是决定性的

在 `beam=16` 的剩余 miss 子集上：

- `hint_1` 可恢复约 `50.15%`

这说明：

- 第一位 prefix 确实重要
- 但很多样本并不是只差第一位

### 3.3 `hint_2` 是真正的拐点

在 `hint_1` 仍失败的剩余子集上：

- `hint_2` 还能恢复约 `98.74%`

这说明：

- 任务的 hardest routing 大多集中在前两位 token
- 一旦前两位确定，后面的 suffix 对绝大多数样本已经不难

### 3.4 `hint_3` 更像分析上界

`hint_3` 后基本只剩极少量 residual hard set。

这说明：

- `hint_3` 很强
- 更接近“把答案前缀几乎直接给到只剩末位判别”
- 它适合作为分析上界，但是否适合作为正式训练信号，需要谨慎

### 3.5 residual hard set 高度集中

最终 `hint_3` 仍失败的样本，不是随机散布，而是集中在少数特定 prefix bucket。

这说明：

- 最后剩下的问题不是“prefix 都不会”
- 而更像：
  - 某些特定 bucket 下最后一级 token 判别极难
  - 或者 codebook 最后一级 collision 很强

---

## 4. 本地环境与服务器环境

这是当前工程状态里非常重要的一点。

### 4.1 本地环境

当前本地开发环境：

- 路径形如：
  - `/Users/fanghaotian/Desktop/src/GenRec`
- **没有 GPU**
- 主要用途：
  - 代码阅读
  - 分析脚本编写
  - 单元测试
  - `--dry-run`
  - notebook / 文档

因此本地环境**不能**作为真实训练验证环境。

### 4.2 服务器环境

当前真实训练环境：

- 路径形如：
  - `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec`
- **有 GPU**
- 用于：
  - `accelerate launch`
  - multi-GPU / DeepSpeed 训练
  - 大规模 beam search

因此：

- 本地改动是否“逻辑正确”，需要靠本地测试验证
- 但是否“训练真的能跑起来”，最终要看服务器环境

### 4.3 路径差异的含义

这意味着后续所有脚本设计都必须注意：

- 本地默认值与服务器默认值可能不同
- 不能依赖单一绝对路径
- 需要支持：
  - `REPO_ROOT`
  - `MODEL_PATH`
  - `DATA_DIR`
  - `INDEX_PATH`
  - `DS_CONFIG`
  - `LOG_DIR`
  等可覆盖参数

---

## 5. 当前已经完成的代码改动

### 5.1 离线分析侧

文件：

- `GenRec/analyze_rl_beam_hint.py`

新增能力：

- 支持读取 `beam=16` cascade `details`
- 支持从 `details` 导出：
  - `extra_info.index -> hint_depth`
- 导出的 fixed hint map 结构大致为：

```json
{
  "sample_key_type": "extra_info.index",
  "beam_size": 16,
  "default_unsolved_depth": 3,
  "max_available_stage_depth": 3,
  "hint_depth_by_index": {
    "123": 0,
    "124": 1,
    "125": 2,
    "126": 3
  },
  "unsolved_indices": [126]
}
```

这个 map 是后续 fixed-hint 训练的输入。

### 5.2 训练输入侧

文件：

- `GenRec/fixed_hint_utils.py`

新增能力：

- 从 fixed hint map 给样本注入：
  - `oracle_hint_depth`
  - `oracle_hint_text`
  - `oracle_hint_unsolved`

### 5.3 reward 侧

文件：

- `GenRec/rewards/ranking_reward.py`

新增逻辑：

- 如果存在 `oracle_hint_text`
- 那么 reward 计算时会把：
  - `hint_text + generated_suffix`
 还原成完整 completion，再算：
  - `rule_reward`
  - `prefix_rule_reward`
  - `ndcg_rule_reward`

其含义是：

- prompt 里虽然已经给了 hint prefix
- 但 reward 仍然针对**完整 item**算，而不是只看 suffix

### 5.4 processor 侧

文件：

- `GenRec/logit_processor.py`
- `GenRec/fixed_hint_logit_processor.py`
- `GenRec/util.py`

当前设计是：

- 默认旧实验继续走：
  - `logit_processor.py`
- fixed-hint 路径单独走：
  - `fixed_hint_logit_processor.py`

这样做的目的是：

- 避免新逻辑污染已有实验

### 5.5 trainer 侧

文件：

- `GenRec/trl_trainer.py`
- `GenRec/fixed_hint_grpo_trainer.py`

当前逻辑：

- 如果没有 `fixed_hint_depth_map_path`
  - 继续走普通 trainer 路径
- 如果有 `fixed_hint_depth_map_path`
  - 切到 `FixedHintRuleOnlyGRPOTrainer`

### 5.6 一键启动脚本

文件：

- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint.sh`

这个脚本当前负责：

1. 调 analyzer 导出 fixed hint map
2. 再直接 `accelerate launch trl_trainer.py`

也就是说：

- 它已经不再调用内部旧版 `rule-only.sh`
- 是一个自包含 launcher

---

## 6. 当前真正遇到的工程问题

这是当前文档里最关键的部分。

### 6.1 问题现象

服务器上启动 fixed-hint `rule_only` RL 后：

- 预处理阶段能正常完成：
  - fixed hint map 导出成功
  - train/eval dataset 注入 `oracle_hint_depth` 成功
  - model / optimizer / deepspeed 初始化成功
- 但训练在第一个 step 进入 generation 阶段后挂住 / 崩溃

### 6.2 最初误判

一开始曾经有一个误判：

- 以为问题在于：
  - 不同 hint depth 的样本在同一个 sample group 里一起做 reward normalization

这个判断**不准确**。

因为：

- GRPO 的 reward / advantage 归一化本来就是按 sample 的 `num_generations` group 做的
- 不是把整个 training batch 所有 sample 混在一起归一化

因此，问题**不在 reward normalization 层**。

### 6.3 当前更准确的根因判断

当前更准确的判断是：

> 问题出在 **generation 阶段的分布式同步**。

更具体地说：

- `FixedHintRuleOnlyGRPOTrainer` 当前实现里，会先按本地 batch 的 `oracle_hint_depth` 做分桶
- 然后对每个 depth bucket 单独调用一次：
  - `self._generate(...)`

问题在于：

- `_generate(...)` 不是一个纯本地函数
- 它内部会做：
  - token 数统计
  - length 统计
  - `accelerator.gather(...)`
  - 其它跨 rank 的同步操作

但各个 rank 本地的 depth bucket 分布可能不同：

- rank0 可能有 `d=0,1,2`
- rank1 可能有 `d=0,2,3`
- rank2 可能有 `d=1,2`
- rank3 可能有 `d=0,1,3`

于是就会出现：

- 不同 rank 调用 `_generate(...)` 的次数不同
- 调用顺序不同
- 每次调用里的 bucket size 也不同

在分布式训练里，这会导致：

- 某个 rank 已经进入下一次 generate 的 gather
- 另一个 rank 还停留在上一轮
- 最终表现成 hang / deadlock / 或某个 rank 上 assertion 爆炸

### 6.4 为什么 analyzer 不会出这个问题

离线 analyzer：

- 本质上也是按 `hint_depth` 分 bucket generate

但 analyzer 跑的时候：

- 不是一个正在做 distributed RL 训练的 `GRPOTrainer` step
- 而是一个独立分析脚本
- 它的调用路径、同步方式和 TRL 训练时不同

所以：

- analyzer 能跑
- 不推出相同分桶逻辑在 TRL 多卡训练里也能直接跑

### 6.5 当前报错的表面位置

日志里最终爆在：

- `fixed_hint_logit_processor.py`
- `No valid tokens for prefix_ids`

但这更像是：

- 分布式 generate 状态已经不一致之后出现的表面症状

而不是最根本的问题。

---

## 7. 当前实现路线的分歧

到现在为止，固定 oracle hint depth 的训练实现，出现了两条路线。

### 路线 A：训练时动态按 depth 分桶

当前 `FixedHintRuleOnlyGRPOTrainer` 就是这条路线。

优点：

- 逻辑上最接近“sample-wise oracle depth”
- 一个训练 job 内可以同时覆盖：
  - `d=0`
  - `d=1`
  - `d=2`
  - `d=3`

缺点：

- 在多卡分布式环境下，generate 同步非常脆弱
- 容易因为不同 rank 的 bucket 分布不同而挂住

### 路线 B：离线预处理 prompt

也就是：

- 先根据 fixed hint map 把 hint 直接拼到 prompt 里
- 训练时不再 runtime 拼接

这个方向能解决的问题：

- 不再需要训练时查 map
- 不再需要训练时拼 prompt

但它**不能单独解决**的问题：

- 如果预处理后的数据仍然把 `d=0/1/2/3` 样本混在一个 generation step 里，
  那 generation 阶段的异构性仍然存在

所以：

- 预处理 prompt 本身是有价值的
- 但不是当前 deadlock 的完整解法

### 路线 C：按 depth 分开的预处理数据

也就是预处理成多份数据：

- `train_hint_d0.json`
- `train_hint_d1.json`
- `train_hint_d2.json`
- `train_hint_d3.json`

这样每次训练只使用单一 depth 数据。

优点：

- generation 状态同质
- 最容易在多卡下稳定运行
- 代码实现最简单

缺点：

- 不再是“一个 job 内 sample-wise oracle mixed training”
- 更像 4 个条件实验，或 4 个阶段数据源

---

## 8. 当前对这些路线的理解

### 8.1 不是“不能多次 generate”

有一个非常重要的澄清：

- 问题**不是**“一个 step 里不能多次 generate”
- 很多 reasoning / multi-rollout 系统本来就会多次 generate

真正的问题是：

> **这些 generate 调用在所有 rank 上是否严格同步。**

所以：

- “多次 generate”本身不是错
- “按本地动态 bucket 多次 generate”才是当前实现的危险点

### 8.2 如果要保留单 step 内多次 generate

那么必须做到：

- 所有 rank 固定遍历同样的 depth 序列：
  - `0 -> 1 -> 2 -> 3`
- 每个 rank 每个 depth 都必须调用一次 generate
- 哪怕某个 rank 某个 depth 没样本，也要走占位逻辑

这条路不是做不到，但实现复杂度明显更高。

### 8.3 如果目标是先把实验跑起来

那么最稳的是：

- 不要在 training step 内做本地动态分桶 generate
- 而是让训练数据本身按 depth 分开

也就是：

- 工程上更保守
- 但更可控

---

## 9. 当前对 fixed hint 训练实验语义的理解

即使工程问题解决了，这个实验本身也有一个重要语义限制。

当前 fixed-hint `rule_only` RL 实验，本质上是在做：

> **Oracle-scaffolded rule-only RL**

它最直接回答的是：

- 如果我们给每个样本一个几乎 oracle 的 prefix scaffold，`rule_only` RL 会不会更容易拿到正 reward

它不直接回答的是：

- 模型有没有因此学会无 hint 条件下自己生成这些 prefix

因为：

- hint token 是 external scaffold
- 不是模型自己在原始 prompt 下生成的 action

所以这个实验很有价值，但定位必须准：

- 它更像是“upper-bound scaffold experiment”
- 不是 prefix acquisition 的直接证明

---

## 10. 当前代码状态的判断

### 10.1 必要改动

当前已经加入、且对 fixed-hint 实验确实必要的部分包括：

- fixed hint depth map 导出
- reward 侧 full completion 还原
- fixed hint 专用 processor
- fixed hint 专用 trainer
- fixed hint 一键 launcher

### 10.2 兼容性处理

为了避免污染旧实验，已经做了这些隔离：

- 默认旧实验继续走原始 `logit_processor.py`
- fixed hint 路径单独走 `fixed_hint_logit_processor.py`
- fixed hint launcher 不再调用内部旧版 `rule-only.sh`

### 10.3 当前最不稳定的部分

当前最不稳定、也是最可能要进一步重构的部分，是：

- `FixedHintRuleOnlyGRPOTrainer` 里按本地 `hint_depth` 分桶后调用 `_generate()`

这一块是目前工程实现中的最大风险点。

---

## 11. 当前文档想保留的判断

到目前为止，最重要的判断可以总结为：

- 离线 analyzer 的结论很强：
  - 前两位 prefix token 才是主要难点
  - `d=2` 是真正的强 scaffold
  - `d=3` 更像分析上界
- 固定 oracle hint depth 作为训练实验是合理的研究问题
- 但 sample-wise mixed-depth 训练在当前 TRL 多卡 generate 路径下并不稳定
- 如果优先目标是让实验先稳定跑起来，那么：
  - **把 runtime 动态分桶转成离线结构化数据，是更稳的工程方向**

---

## 12. 当前希望这份文档发挥的作用

这份文档不是为了宣布“方案已经定了”，而是为了让后续任何继续接手这个方向的人快速知道：

- 我们已经知道了什么
- 现在失败到底失败在哪
- 哪些问题是研究问题
- 哪些问题只是工程同步问题

这样就能避免后面再重复：

- 把 reward normalization 当成根因
- 把 analyzer 能跑误以为 trainer 也一定能跑
- 把“预处理 prompt”误当成 deadlock 的完整解法

这就是当前这份状态记录的目的。

