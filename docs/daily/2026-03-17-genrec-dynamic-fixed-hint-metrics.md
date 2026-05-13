# GenRec Dynamic Hint / Fixed Hint 指标与实现说明

本文说明两件事：

1. `dynamic hint` 训练里新增的指标分别表示什么，应该怎么读。
2. `fixed hint` 与 `dynamic hint` 在代码里分别是如何实现的，它们的核心区别是什么。

相关代码入口：

- `GenRec/trl_trainer.py`
- `GenRec/fixed_hint_utils.py`
- `GenRec/fixed_hint_grpo_trainer.py`
- `GenRec/fixed_hint_logit_processor.py`
- `GenRec/rewards/ranking_reward.py`
- `GenRec/analyze_rl_beam_hint.py`

## 1. 先看核心区别

两条路径都会把一段 oracle SID prefix hint 拼到 prompt 后面，再做 constrained generation；差别在于 hint depth 是什么时候决定的。

| 模式 | hint depth 的来源 | 决策时机 | 一次 batch 会不会多次生成 | 适合回答的问题 |
| --- | --- | --- | --- | --- |
| `fixed hint` | 离线分析得到的 `hint_depth_by_index` | 训练前，数据注入时 | 不会 | “这个样本的 oracle scaffold 深度预先设成多少？” |
| `dynamic hint` | 当前策略在线 rollout 的 stage-wise 结果 | 训练时，生成过程中 | 会，按 stage 递增 | “当前策略最少需要多深的 hint 才能打中 rule？” |

可以把两者理解成：

- `fixed hint` 是先查表，再生成一次。
- `dynamic hint` 是先不加 hint 试一次，不行再逐层加 hint 继续试，直到命中或者到达上限。

## 2. Fixed Hint 是怎么实现的

### 2.1 离线产物

`fixed hint` 依赖 `analyze_rl_beam_hint.py` 导出的 depth map。  
`build_fixed_hint_depth_map_from_details(...)` 会从离线 cascade 结果中生成一个 JSON，大致包含：

- `hint_depth_by_index`: `extra_info.index -> hint depth`
- `unsolved_indices`: 到最后一个 stage 仍未命中的样本
- `default_unsolved_depth`: 对未命中样本使用的默认深度
- `max_available_stage_depth`: 离线分析里最大的可用 stage depth

也就是说，`fixed hint` 不是在线决定 hint，而是把“这个样本应该给几层 hint”提前算好。

### 2.2 数据注入

`trl_trainer.py` 在检测到 `fixed_hint_depth_map_path` 后，会：

1. 读入固定 hint map。
2. 对 train dataset 调 `apply_fixed_hint_depth_to_example(...)`。
3. 给每个样本补三个字段：
   - `oracle_hint_depth`
   - `oracle_hint_text`
   - `oracle_hint_unsolved`

`oracle_hint_text` 由 `fixed_hint_utils.py` 里的 `build_hint_text(...)` 构造，本质上就是从 `reward_model.ground_truth` 中取前 `hint_depth` 个 SID token 拼接起来。

如果样本在 `unsolved_indices` 里：

- 会使用 `fixed_hint_unsolved_depth` 或 map 里的 `default_unsolved_depth`
- 然后再受 `fixed_hint_depth_cap` 约束

### 2.3 Prompt 构造与生成

`FixedHintRuleOnlyGRPOTrainer` 在 `fixed_hint_grpo_trainer.py` 中实现。

它会调用 `build_prompt_with_hint(...)`，把：

- 聊天模板后的 prompt
- `oracle_hint_text`

直接拼起来，然后对整个 batch 做一次 `_generate(...)`。

这里的关键点是：`fixed hint` 的本地 batch 在进入生成前，hint depth 已经是确定的，所以它只做单次 rollout，不做 stage 递增。

### 2.4 为什么要用 `FixedHintConstrainedLogitsProcessor`

`fixed_hint_logit_processor.py` 里的 `FixedHintConstrainedLogitsProcessor` 会在当前序列里重新定位有效的 `prefix_ids` 起点，再调用 trie 的 `prefix_allowed_tokens_fn(...)`。

它和普通 constrained processor 的区别在于：

- prompt 后面现在真的多了一段 hint token
- 所以不能再假设“生成约束前缀一定紧挨着原始 prompt 末尾”
- 必须在“带 hint 的当前序列”里重新找合法的 trie continuation

因此只要 prompt 里包含 oracle hint，固定 hint 和动态 hint 都要走这条 processor 路径。

### 2.5 Reward 怎么算

`rewards/ranking_reward.py` 的 `rule_reward(...)` 和 prefix reward 都不会只看模型新生成的 suffix。

如果 reward kwargs 里带有 `oracle_hint_text`，它们会先重建：

- `full_completion = oracle_hint_text + generated_suffix`

然后再和 ground truth 比较。

这很关键，因为 hint 是 oracle prefix，本身就是目标序列的一部分。  
如果不把 hint prepend 回去，reward 会错误地把“模型其实命中了完整序列”的样本判成没命中。

## 3. Dynamic Hint 是怎么实现的

### 3.1 启用条件

`trl_trainer.py` 中只要：

- `dynamic_hint_max_depth` 不为空
- 且 `dynamic_hint_max_depth > 0`

就会启用 `dynamic hint` 路径，并切到 `DynamicHintRuleOnlyGRPOTrainer`。

当前实现还显式限制：

- `dynamic hint` 只支持 `reward_mode=rule_only`

原因是现在的在线停止条件是“某个 stage 是否出现了至少一个 rule hit”，这个定义只和 `rule_only` 是完全对齐的。

### 3.2 Runtime cascade 的 batch 组织方式

`dynamic hint` 不是按 completion 单条推进，而是按 prompt group 推进。

一个 group 的大小等于：

- `num_generations`

例如你现在的配置里 `num_generations=16`，那么一个原始 prompt 对应 16 条 completion，它们一起构成一个 group。

后续所有 stage 的决策单位都是 group，而不是单条 completion：

- 只要该 group 在某个 stage 里“16 条 completion 里至少有 1 条 rule hit”
- 这个 group 就视为在该 stage 被解决
- 后续 stage 不再继续尝试这个 group

### 3.3 每个 stage 实际做了什么

`DynamicHintRuleOnlyGRPOTrainer._run_dynamic_hint_cascade(...)` 的流程是：

1. 先把输入 batch 切成若干个 prompt group。
2. 初始时所有 group 都放在 `unresolved_groups` 里。
3. 对 `requested_hint_depth = 0, 1, 2, ..., max_hint_depth` 依次循环：
   - 只对当前 unresolved groups 构造 stage 输入
   - 用 `_build_runtime_hinted_example(...)` 给每个样本即时写入：
     - `oracle_hint_depth`
     - `oracle_hint_text`
     - `oracle_hint_unsolved=False`
   - 用 `_build_hinted_prompts(...)` 生成 stage prompt
   - 用 `_generate_single_turn(...)` 做这一 stage 的生成
   - 解码 completion
   - 用 `_compute_rule_hit_rewards(...)` 判断该 stage 下每个 completion 是否命中完整规则
4. 对每个 group：
   - 如果该 stage 任意一条 completion 命中，则选择当前 stage 输出，group 停止升级
   - 如果没命中且还没到最大深度，则进入下一 stage
   - 如果已经到最大深度，即使没命中也强制选择这一 stage 的输出

最终，每个原始 prompt group 都会恰好选中一个 stage 的 16 条 completion。

### 3.4 `hint_depth` 的记录方式

当前实现只保留一套 `hint_depth`：

- 它就是当前 stage 请求并最终被选中的 hint 深度

在我们当前的数据设定里，ground truth 长度是对齐的，因此不再单独区分：

- `requested_hint_depth`
- `actual_hint_depth`

所以后面的 dynamic hint 指标里，只要看到某个样本或 group 的 `hint_depth=k`，就直接理解为“它最终选中了 stage `k`”即可。

### 3.5 选中 stage 后如何进入 GRPO

当 cascade 结束后，代码会把每个 group 选中的结果重新按原始顺序拼回去，得到：

- `selected_inputs`
- `selected_prompt_ids_list`
- `selected_completion_ids_list`
- `selected_completions_text`

然后后面的流程和普通 GRPO 基本一致：

1. pad `prompt_ids` / `completion_ids`
2. 计算 `old_per_token_logps` / `ref_per_token_logps`
3. 调 reward 函数
4. 算 grouped rewards、advantages、KL、entropy 等

也就是说：

- `dynamic hint` 的特殊性只在“如何先选出最终那一组 completion”
- 一旦选完，后续 reward / advantage / trainer logging 都是基于最终选中的那组结果

### 3.6 Eval 阶段的行为

`DynamicHintRuleOnlyGRPOTrainer._generate_and_score_completions(...)` 里有：

- train 阶段：`max_hint_depth = self.dynamic_hint_max_depth`
- eval 阶段：
  - 如果 `dynamic_hint_apply_to_eval=True`，也用同样的 cascade
  - 如果 `False`，则 `max_hint_depth=0`

所以当 `dynamic_hint_apply_to_eval=False` 时，eval 实际等价于只跑 base stage，不会做 hint 递增。

## 4. Dynamic Hint 新增指标都是什么意思

下面这些都是 `fixed_hint_grpo_trainer.py::_log_dynamic_hint_metrics(...)` 里新增的。

注意它们的统计单位是“prompt group”，不是单条 completion。  
如果 `num_generations=16`，那一个 group 就是一条原始样本对应的 16 条生成结果。

### 4.1 `dynamic_hint/selected_hint_depth_mean`

含义：

- 所有 group 最终选中 stage 的平均 hint 深度

公式：

- `sum(selected_group_hint_depths) / num_groups`

解释：

- 越低说明当前策略越不依赖 hint
- 越高说明需要更深的 oracle scaffold 才能打中规则

### 4.2 `dynamic_hint/selected_depth_{k}_frac`

含义：

- 最终选中的 stage 深度等于 `k` 的 group 占比

解释：

- `selected_depth_0_frac` 高，说明大量样本 base stage 就能解决
- `selected_depth_2_frac` 高，说明很多样本要到 depth 2 才首次出现 rule hit
- 对最大深度来说，这个值里既包含“在最大深度才命中”的 group，也包含“到最大深度依然不命中、但被强制保留”的 group

### 4.3 `dynamic_hint/max_depth_miss_frac`

含义：

- 到达 `max_hint_depth` 后仍然没有任何 rule hit 的 group 占比

代码定义是：

- `hint_depth == max_hint_depth`
- 且 `rule_hit_any == False`

解释：

- 这是“cascade 已经走到头了，但还是没解决”的比例
- 如果它偏高，通常说明：
  - 当前模型即使拿到最大 hint 仍不会做
  - 或者 `max_hint_depth` 设得太浅

### 4.4 `dynamic_hint/stage_{k}_rule_hit_frac`

含义：

- 进入 stage `k` 的 group 中，有多少比例在 stage `k` 首次出现了至少一个 rule hit

分母不是全 batch，而是：

- `evaluated_group_count`
- 也就是真正进入这一 stage 的 unresolved groups 数量

解释：

- 这是“这一层 hint 本身的有效率”
- 它回答的是：对那些前面还没解决的样本来说，给到当前深度的 hint 后，有多少能被救回来

### 4.5 `dynamic_hint/stage_{k}_remaining_frac`

含义：

- 进入 stage `k` 的 group 中，有多少比例在这一层之后仍未解决，需要继续进入下一层

解释：

- 这个值越高，说明当前 stage 的 hint 强度不够
- 在最大 stage 上它一定是 `0.0`

### 4.6 `_logs["dynamic_hint_depth"]`

这不是 console metric，而是 trainer 内部额外保存的一份日志：

- 每个 group 最终选中的实际 hint 深度列表

它更适合做后处理分析，例如和离线 fixed hint depth map 做分布对比。

## 5. 怎么读一条真实日志

假设日志里有：

```text
dynamic_hint/selected_hint_depth_mean = 1.25
dynamic_hint/selected_depth_0_frac = 0.25
dynamic_hint/selected_depth_1_frac = 0.25
dynamic_hint/selected_depth_2_frac = 0.5
dynamic_hint/max_depth_miss_frac = 0.0
dynamic_hint/stage_0_rule_hit_frac = 0.25
dynamic_hint/stage_1_rule_hit_frac = 0.3333333333333333
dynamic_hint/stage_2_rule_hit_frac = 1.0
```

如果这一步共有 16 个 prompt groups，那么它表示：

- 4 个 group 在不加 hint 时就命中了
- 剩余 12 个 group 进入 stage 1，其中 4 个被 depth 1 救回来
- 剩余 8 个 group 进入 stage 2，其中 8 个全部被 depth 2 救回来
- 没有 group 一直 miss 到最大深度

所以最终分布就是：

- depth 0: 4 / 16 = 0.25
- depth 1: 4 / 16 = 0.25
- depth 2: 8 / 16 = 0.5

平均深度就是：

- `(0*4 + 1*4 + 2*8) / 16 = 1.25`

而：

- `stage_1_rule_hit_frac = 4 / 12 = 0.3333`

这里的分母是进入 stage 1 的 12 个 group，不是全体 16 个 group。

## 6. Dynamic / Fixed Hint 和其他训练指标的关系

dynamic hint 相关指标只负责描述“最终选中的 stage 是怎么来的”。  
其它常见指标，例如：

- `completions/mean_length`
- `completions/clipped_ratio`
- `rewards/rule_reward/mean`
- `reward`
- `reward_std`
- `kl`
- `entropy`

它们在 dynamic hint 模式下统计的都是：

- cascade 结束后最终被选中的那一组 completion

而不是把所有 stage 的中间 completion 全部混在一起统计。

所以如果你看到：

- `dynamic_hint/selected_depth_2_frac` 很高
- 同时 `rewards/rule_reward/mean` 也提升了

可以理解为：

- 当前模型虽然 base stage 还不太行
- 但在 depth 2 scaffold 下已经能较稳定地产生 rule hit

## 7. 一句话总结

- `fixed hint` 是离线先决定每个样本该给多深 hint，再单次生成。
- `dynamic hint` 是在线逐层尝试 hint depth，按当前策略 rollout 结果决定最终选哪个 stage。
- `dynamic_hint/*` 指标描述的不是 token-level 行为，而是“一个 prompt group 在 cascade 中被哪一层 hint 解决”的统计。
