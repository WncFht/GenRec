# GenRec RL 脚本 reward 形式详解（2026-03-13）

本文档专门解释 `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/` 这组 RL 脚本当前各自对应的 reward 形式。

目标不是泛泛介绍 “有哪几种 reward”，而是把下面 3 件事说清楚：

1. **每个脚本到底优化什么 reward**。
2. **reward 是 sequence-level 还是 token-level**。
3. **给定同一个 ground truth 和同一组 rollout，不同脚本会得到什么 reward / advantage 形状**。

本文以当前仓库 `HEAD` 为准，核心实现文件是：

- `GenRec/rewards/ranking_reward.py`
- `GenRec/token_prefix_grpo_trainer.py`
- `GenRec/trl_trainer.py`
- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/RL_PRESETS.md`

---

## 1. 先说结论

当前这组脚本一共可以归到 6 类 reward 行为：

1. **`rule_only`**
   - 只有 exact match 才给 `1.0`，否则 `0.0`。
   - 是最稀疏的 reward。

2. **`prefix_seq_only`**
   - 用 sequence-level prefix reward：`matched_prefix_len / gt_len`。
   - 只看“前缀对了几位”，但 reward 仍是 rollout 级标量。

3. **`prefix_token_only`**
   - 用 token-level prefix reward。
   - 当前默认不是 `1/0`，而是**每个 matched token 给 `1 / gt_len`**。
   - 然后按 **token 位置** 做 group normalization。

4. **`prefix_token` / `prefix`**
   - 同时使用 prefix + ndcg。
   - prefix 部分先做 sequence-level normalization，再把 advantage 均匀分给 matched tokens。
   - ndcg 部分是 sequence-level，再广播到所有 completion token。

5. **`prefix_token_totalnorm`**
   - 也是 prefix + ndcg。
   - 但它不是“先 sequence 再分发”，而是**先在 token 上合成 raw reward，再按 token 位置做 normalization**。

6. **`prefix_token_totalnorm_errtok`**
   - 与上一种类似。
   - 区别是 ndcg 惩罚不再广播到所有 token，而是**只打到 error tokens**（第一个错误及其后面那些 token）。

另外，**当前版本所有非 `rule_only` 模式都会记录 `rule_reward`**，但大多数情况下它只是一个 **zero-weight probe**，用于日志观察，不参与优化。

---

## 2. 当前代码里的 3 个基础 reward

## 2.1 `rule_reward`

定义在 `GenRec/rewards/ranking_reward.py`。

含义最简单：

- completion 与 ground truth 完全一致：`1.0`
- 否则：`0.0`

公式可以写成：

```text
rule_reward = 1[completion == ground_truth]
```

它是**sequence-level scalar reward**。

---

## 2.2 `prefix_rule_reward`

也是定义在 `GenRec/rewards/ranking_reward.py`。

先看 completion 和 ground truth 的**最长公共前缀长度** `matched_prefix_len`：

```text
gt       = [g1, g2, g3, g4]
pred     = [p1, p2, p3, p4]

matched_prefix_len = 从第 1 位开始连续相等，直到遇到第一个不等为止
```

当前脚本默认 `PREFIX_REWARD_NORMALIZE=true`，所以 sequence-level prefix reward 是：

```text
prefix_rule_reward = matched_prefix_len / len(gt)
```

如果把 `PREFIX_REWARD_NORMALIZE=false`，则会变成：

```text
prefix_rule_reward = matched_prefix_len
```

注意这里是**连续前缀**，不是“整条序列一共对了几个”。

例如：

- `[10, 20, 30, 40]` vs `[10, 20, 30, 99]` -> `matched_prefix_len = 3`
- `[10, 20, 30, 40]` vs `[10, 99, 30, 40]` -> `matched_prefix_len = 1`
- `[10, 20, 30, 40]` vs `[99, 20, 30, 40]` -> `matched_prefix_len = 0`

---

## 2.3 `ndcg_rule_reward`

也定义在 `GenRec/rewards/ranking_reward.py`。

它不是看 token 对不对，而是看**当前 rollout 在组内的 rank 位置**。

对 `num_beams = 4` 的一组 rollout，代码会先构造一组负值：

```text
[-0.3904, -0.2463, -0.1952, -0.1681]
```

几个关键点：

1. **它跟 rollout 在 group 里的顺序绑定**，而不是和 matched token 个数绑定。
2. **只要该 group 里至少有一个 exact match，非 exact rollout 才会拿到这些负值**。
3. **如果整组一个 exact match 都没有，整组 ndcg reward 都是 0**。

也就是说：

- “exact hit 存在” -> 非 exact rollout 拿 rank-based negative reward
- “exact hit 不存在” -> 整组 ndcg 全部置 0

---

## 3. token-level prefix reward 在当前代码里的真实定义

token-level prefix reward 不是直接由 `ranking_reward.py` 返回，而是 `TokenPrefixGRPOTrainer` 在 `GenRec/token_prefix_grpo_trainer.py` 里现算的。

在当前默认配置 `PREFIX_REWARD_NORMALIZE=true` 下，若 ground truth 有 `L` 个有效 SID token，则：

```text
prefix_token_reward[t] =
  1 / L    if 第 t 个 token 属于 matched prefix
  0        otherwise
```

因此，对一个 4-token ground truth：

- prefix 长度为 4 -> `[0.25, 0.25, 0.25, 0.25]`
- prefix 长度为 3 -> `[0.25, 0.25, 0.25, 0.00]`
- prefix 长度为 1 -> `[0.25, 0.00, 0.00, 0.00]`
- prefix 长度为 0 -> `[0.00, 0.00, 0.00, 0.00]`

如果以后把 `PREFIX_REWARD_NORMALIZE=false`，它才会变成你更直觉的 `1/0` 形式：

- prefix 长度为 4 -> `[1, 1, 1, 1]`
- prefix 长度为 3 -> `[1, 1, 1, 0]`
- prefix 长度为 1 -> `[1, 0, 0, 0]`
- prefix 长度为 0 -> `[0, 0, 0, 0]`

但要注意：**在 `token_adv_total_token_normalize=true` 的分支里，把 `0.25/0` 改成 `1/0` 往往不会改变归一化后的相对模式**，因为同一列上只是整体乘了常数。

---

## 4. 当前脚本与 reward 形式总表

下表聚焦 reward 行为，不重复列无关训练参数。

| 脚本 | 等价 preset / 模式 | 优化的 reward | 训练信号粒度 | 备注 |
|---|---|---|---|---|
| `...-rl-rule-only.sh` | `rule_only` | `rule_reward` | sequence-level | 最稀疏，只看 exact match |
| `...-rl-prefix-seq-only.sh` | `prefix_seq_only` / `prefix_rule_only + token_level_prefix_adv=false` | `prefix_rule_reward` | sequence-level | `rule_reward` 仅记录 |
| `...-rl-prefix-token-only.sh` | `prefix_token_only` / `prefix_rule_only + token_level_prefix_adv=true + totalnorm=true` | token-level prefix reward | token-level | 当前默认每个 matched token 给 `1/L` |
| `...-rl-prefix-token.sh` | `prefix_token` / `prefix_only + token_level_prefix_adv=true + totalnorm=false` | prefix + ndcg | mixed | prefix 先 sequence-normalize，再分配到 matched tokens；ndcg 作为 sequence signal 广播到所有 token |
| `...-rl-prefix.sh` | 与 `...-rl-prefix-token.sh` 相同 | prefix + ndcg | mixed | reward 行为相同，可视为旧别名 |
| `...-rl-prefix-token-totalnorm.sh` | `prefix_token_totalnorm` | prefix + ndcg | token-level | 先在 token 上合成 raw reward，再做 token-wise normalization |
| `...-rl-prefix-token-totalnorm-errtok.sh` | `prefix_token_totalnorm_errtok` | prefix + ndcg(error-only) | token-level | ndcg 惩罚只打到错误 token 上 |
| `...-rl.sh` | 统一入口 | 取决于 `--preset` | 取决于 preset | 推荐用它替代旧脚本 |

补充说明：

- `...-rl-prefix-token.sh` 和 `...-rl-prefix.sh` 当前 reward 相关默认值完全一致。
- 尽管旧脚本里还保留了 `PROBE_RULE_ZERO_WEIGHT` 变量，当前 `HEAD` 实现里，非 `rule_only` 模式会统一把 `rule_reward` 作为 zero-weight probe 记录下来。

---

## 5. 例子 A：同一组 rollout，在不同脚本下会得到什么 reward

为了把差异讲清楚，下面固定一个最小示例。

## 5.1 Ground truth 与 rollout 组

令 ground truth 为：

```text
GT = [10, 20, 30, 40]
```

等价的 SID 字符串可以理解成：

```text
<a_10><b_20><c_30><d_40>
```

设一个 beam group（`num_beams = 4`）的 rollout 依次为：

| rollout | completion | matched prefix len | 是否 exact |
|---|---|---:|---:|
| R1 | `[10, 20, 30, 40]` | 4 | 是 |
| R2 | `[10, 20, 30, 99]` | 3 | 否 |
| R3 | `[10, 99, 88, 77]` | 1 | 否 |
| R4 | `[99, 88, 77, 66]` | 0 | 否 |

在当前默认 `PREFIX_REWARD_NORMALIZE=true` 下，3 个基础 reward 的 raw 输出分别是：

| rollout | `rule_reward` | `prefix_rule_reward` | `ndcg_rule_reward` |
|---|---:|---:|---:|
| R1 | `1.0000` | `1.0000` | `0.0000` |
| R2 | `0.0000` | `0.7500` | `-0.2463` |
| R3 | `0.0000` | `0.2500` | `-0.1952` |
| R4 | `0.0000` | `0.0000` | `-0.1681` |

这里的 `ndcg_rule_reward` 之所以不是全 0，是因为这组里存在一个 exact hit（R1）。

---

## 5.2 `rule_only`

对应脚本：

- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh`

它只优化：

```text
[1.0, 0.0, 0.0, 0.0]
```

所以它的特征非常明确：

- R1 有正 reward
- R2 / R3 / R4 在 raw reward 层面完全等价
- prefix 对 3 位和 prefix 对 1 位，不会有任何区分

也正因如此，它最容易遇到“整组都没有有效 advantage”的问题。

---

## 5.3 `prefix_seq_only`

对应脚本：

- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-prefix-seq-only.sh`

它优化的是 sequence-level prefix reward：

```text
[1.00, 0.75, 0.25, 0.00]
```

同时 `rule_reward` 会被记录为 probe：

```text
[1.00, 0.00, 0.00, 0.00]
```

这一模式的核心特征是：

- R2 比 R3 更好，因为 prefix 更长。
- 但 reward 仍是 **rollout 级标量**。
- 换句话说，它知道 “R2 比 R3 好”，但**不知道 R2 的第 1/2/3 个 token 各自该拿多少 credit**。

如果你只关心“这一整条 beam 的 prefix 质量”，这是最直接的脚本。

---

## 5.4 `prefix_token_only`

对应脚本：

- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-prefix-token-only.sh`

它的 raw token-level prefix reward 是：

| rollout | raw token reward |
|---|---|
| R1 | `[0.25, 0.25, 0.25, 0.25]` |
| R2 | `[0.25, 0.25, 0.25, 0.00]` |
| R3 | `[0.25, 0.00, 0.00, 0.00]` |
| R4 | `[0.00, 0.00, 0.00, 0.00]` |

然后它会按 **token 位置** 做 group normalization。

归一化后的 token advantage（按当前 `HEAD` 代码精确计算）为：

| rollout | normalized token advantage |
|---|---|
| R1 | `[0.5749, 0.9968, 0.9968, 1.7247]` |
| R2 | `[0.5749, 0.9968, 0.9968, -0.5749]` |
| R3 | `[0.5749, -0.9968, -0.9968, -0.5749]` |
| R4 | `[-1.7247, -0.9968, -0.9968, -0.5749]` |

这张表很重要，因为它揭示了当前实现的真实语义：

1. **不是简单的 “match token 就拿同样大的最终 advantage”**。
2. 因为做的是 **per-position normalization**，越后面的位置，正样本越稀缺时，正 advantage 往往越大。
3. 所以 R1 的第 4 个 token advantage 最大；R2 因为第 4 位没对，第 4 位反而是负的。

这正是你最近讨论的那个点：

> 当前 `prefix-token-only` 不是“match 了几个 token 就每个 matched token 拿同样的最终强度”，而是“每个位置单独标准化，因此后面的正确 token 可能更值钱”。

---

## 5.5 `prefix_token` / `prefix`

对应脚本：

- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-prefix-token.sh`
- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-prefix.sh`

这两个脚本当前 reward 行为相同。

它们的逻辑不是先在 token 上整体做 normalization，而是：

1. 先对 sequence-level prefix reward 做 group normalization。
2. 再把这个 prefix advantage **均匀分给 matched tokens**。
3. 再把 ndcg 的 sequence advantage **广播到所有 token**。

在例子 A 中：

- prefix sequence reward: `[1.00, 0.75, 0.25, 0.00]`
- ndcg sequence reward: `[0.0000, -0.2463, -0.1952, -0.1681]`

按当前代码计算后，最终 token advantage 是：

| rollout | final token advantage |
|---|---|
| R1 | `[1.7015, 1.7015, 1.7015, 1.7015]` |
| R2 | `[-0.6971, -0.6971, -0.6971, -0.8796]` |
| R3 | `[-0.9484, -0.4008, -0.4008, -0.4008]` |
| R4 | `[-0.1473, -0.1473, -0.1473, -0.1473]` |

它和 `prefix_token_only` 最大的不同在于：

- `prefix_token_only`：按位置归一化，后位更稀缺时可能更大。
- `prefix_token`：prefix 部分先是 sequence advantage，再均匀分配给 matched token；**同一 rollout 内 matched token 的 prefix credit 是均分的**。
- 但这里还叠加了 ndcg 的 sequence penalty，因此整个 rollout 的所有 token 都会一起被推高/拉低。

---

## 5.6 `prefix_token_totalnorm`

对应脚本：

- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-prefix-token-totalnorm.sh`

它先把 prefix token reward 与 ndcg token reward 合并成 raw token reward：

```text
total_token_reward = prefix_token_reward + ndcg_reward_broadcast_to_all_tokens
```

在例子 A 中，合并后的 raw token reward 是：

| rollout | raw token reward |
|---|---|
| R1 | `[0.2500, 0.2500, 0.2500, 0.2500]` |
| R2 | `[0.0037, 0.0037, 0.0037, -0.2463]` |
| R3 | `[0.0548, -0.1952, -0.1952, -0.1952]` |
| R4 | `[-0.1681, -0.1681, -0.1681, -0.1681]` |

再做 token-wise normalization 后得到：

| rollout | normalized token advantage |
|---|---|
| R1 | `[1.4386, 1.5613, 1.5613, 1.7124]` |
| R2 | `[-0.2102, 0.1751, 0.1751, -0.7879]` |
| R3 | `[0.1320, -0.9443, -0.9443, -0.5304]` |
| R4 | `[-1.3604, -0.7920, -0.7920, -0.3941]` |

它和 `prefix_token_only` 的关键区别是：

- 这里不再只看 prefix signal。
- non-exact rollout 会因为 ndcg 而在所有 token 上整体下移。
- 即使某个 token 本身属于 matched prefix，只要这条 rollout 的 ndcg 很差，它的总 reward 仍可能不高。

---

## 5.7 `prefix_token_totalnorm_errtok`

对应脚本：

- `GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-prefix-token-totalnorm-errtok.sh`

这个脚本和上一个的区别在于：

- ndcg 惩罚不再广播到所有 token
- 而是只打在 **error tokens** 上

当前代码里的 error token 定义是：

```text
第一个错误 token 及其后面所有有效 token
```

在例子 A 中，合并后的 raw token reward 变成：

| rollout | raw token reward |
|---|---|
| R1 | `[0.2500, 0.2500, 0.2500, 0.2500]` |
| R2 | `[0.2500, 0.2500, 0.2500, -0.2463]` |
| R3 | `[0.2500, -0.1952, -0.1952, -0.1952]` |
| R4 | `[-0.1681, -0.1681, -0.1681, -0.1681]` |

token-wise normalization 后得到：

| rollout | normalized token advantage |
|---|---|
| R1 | `[0.5765, 0.9980, 0.9980, 1.7124]` |
| R2 | `[0.5765, 0.9980, 0.9980, -0.7879]` |
| R3 | `[0.5765, -1.0605, -1.0605, -0.5304]` |
| R4 | `[-1.7294, -0.9354, -0.9354, -0.3941]` |

这个版本更接近下面的想法：

> prefix 已经对的 token 不应该因为 rollout 后续失败而一并受罚；真正该罚的是错误开始之后那部分 token。

---

## 6. 例子 B：如果整组没有 exact hit，会发生什么

再看一个很重要的例子。

令另一组 rollout 为：

| rollout | completion | matched prefix len | 是否 exact |
|---|---|---:|---:|
| R1 | `[10, 20, 30, 99]` | 3 | 否 |
| R2 | `[10, 20, 88, 77]` | 2 | 否 |
| R3 | `[10, 99, 88, 77]` | 1 | 否 |
| R4 | `[99, 88, 77, 66]` | 0 | 否 |

这时：

- `rule_reward = [0, 0, 0, 0]`
- `prefix_rule_reward = [0.75, 0.50, 0.25, 0.00]`
- `ndcg_rule_reward = [0, 0, 0, 0]`

也就是说：

1. **只要整组没有 exact hit，ndcg 分支就完全失效**。
2. 因此 `prefix_only` 相关脚本在这种 group 上，会退化成“只用 prefix signal”。

举例：

- `prefix_token_only` 的 raw token reward：

```text
R1 [0.25, 0.25, 0.25, 0.00]
R2 [0.25, 0.25, 0.00, 0.00]
R3 [0.25, 0.00, 0.00, 0.00]
R4 [0.00, 0.00, 0.00, 0.00]
```

- `prefix_token_totalnorm` 在这个 group 上会和上面等价，因为 ndcg 全 0。

这也是为什么 prefix reward 对 `rule_only` 的主要补充价值，在于：

> 即使整组没有 exact match，prefix 仍然可以继续提供梯度信号。

---

## 7. 例子 C：如果整组连一个正前缀都没有，`errtok` 会怎样

最后再看 `prefix_token_totalnorm_errtok` 的一个边界情况。

如果一组 rollout 全部都是 prefix 长度 0：

```text
matched_prefix_len = [0, 0, 0, 0]
```

那么当前代码里：

- `prefix_token_reward` 全 0
- `group_has_prefix = false`
- 因此 ndcg error-token penalty 也不会启用

最终这组 token reward 仍然全 0。

这说明 `errtok` 版本的设计不是“只要错了就罚”，而是：

> 先要求这组至少出现过一点正前缀信号，再去把 rank penalty 精确地压到错误 token 上。

---

## 8. 统一入口 `rl.sh` 与旧脚本的对应关系

当前更推荐用统一入口：

```bash
bash GenRec/hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl.sh --preset <name>
```

对应关系如下：

| 统一 preset | 旧脚本 |
|---|---|
| `prefix_token` | `...-rl-prefix-token.sh` |
| `prefix_token_totalnorm` | `...-rl-prefix-token-totalnorm.sh` |
| `prefix_token_totalnorm_errtok` | `...-rl-prefix-token-totalnorm-errtok.sh` |
| `prefix_token_only` | `...-rl-prefix-token-only.sh` |
| `prefix_seq_only` | `...-rl-prefix-seq-only.sh` |
| `rule_only` | `...-rl-rule-only.sh` |

补充：`...-rl-prefix.sh` 在 reward 行为上与 `...-rl-prefix-token.sh` 相同，可以视为旧别名脚本。

---

## 9. 哪些地方最容易看错

这里把最容易混淆的点单独列一下。

### 9.1 `prefix_token_only` 不是 “matched token 最终同权”

当前实现里，它是：

1. 先构造 token reward（默认是 `1/L` 与 `0`）
2. 再按 token 位置做 group normalization

因此后面位置如果正样本更稀有，normalized advantage 反而会更大。

### 9.2 只把 `1/L` 改成 `1`，并不能改变上面这个现象

因为在 `token_adv_total_token_normalize=true` 的分支里，同一列只是整体放大了一个常数，token-wise normalization 后相对结构基本不变。

### 9.3 `ndcg_rule_reward` 不是“按 prefix 质量排序”

它当前是按 **group 内 rollout 的位置** 来分配负值的。

所以如果 beam 顺序本身有变化，ndcg 惩罚也会跟着变。

### 9.4 现在基本都会记录 `rule_reward`

虽然旧脚本还有 `PROBE_RULE_ZERO_WEIGHT` 这个参数名，但当前 `HEAD` 里非 `rule_only` 模式会统一挂上 zero-weight 的 `rule_reward` probe。

---

## 10. 如果以后你想改 reward，应该先改哪一层

如果以后想改 reward 语义，通常有 3 个层次可以动：

1. **raw reward 定义层**
   - `GenRec/rewards/ranking_reward.py`
   - `GenRec/token_prefix_grpo_trainer.py` 里的 `prefix_token_reward` 构造

2. **normalization 层**
   - sequence-level `_group_normalize`
   - token-level `_group_normalize_tokenwise`

3. **credit assignment 层**
   - sequence advantage 如何分发到 matched token
   - ndcg 是广播到全部 token，还是只压到 error token

这三层不要混为一谈。

例如：

- 想从 `1/L` 改成 `1`，这是改 **raw reward 定义层**。
- 想让每个 matched token 最终同权，而不是“后面的 token 更值钱”，这是改 **normalization / credit assignment 层**。
- 想只惩罚错误后缀，不惩罚已经对的 prefix，这是改 **credit assignment 层**。

---

## 11. 最后的实用建议

如果只是想快速知道该选哪个脚本：

- 想要最干净的 exact-match baseline：`rule_only`
- 想要稠密一点但仍保持 rollout 级 prefix 分数：`prefix_seq_only`
- 想要纯 prefix 的 token-level 信号：`prefix_token_only`
- 想要 prefix + ranking，但更接近“先 sequence 再分发”：`prefix_token`
- 想要 prefix + ranking，并且直接在 token 上做总归一化：`prefix_token_totalnorm`
- 想要 prefix + ranking，且只罚错误 token：`prefix_token_totalnorm_errtok`

如果后面要继续讨论“当前 token-level prefix reward 是否该改成 1/0 风格”或者“是否应改成先 sequence normalize 再均匀分发”，建议直接以本文第 5.4 / 5.5 / 5.6 的例子为讨论基准，不容易把不同层次的问题混在一起。
