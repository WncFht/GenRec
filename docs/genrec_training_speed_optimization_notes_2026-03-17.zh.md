# GenRec 训练速度优化笔记（不改变 step 个数）

## 1. 目的

这份文档记录当前 `GenRec` 里 `rule_only + fixed hint / dynamic hint` 训练路径的实现背景，并整理在 **不改变每个 epoch 的 step 个数**、也不改变训练语义的前提下，可以考虑的代码层优化点。

这里的“不改变 step 个数”具体指：

- 不改 train dataset 的样本数
- 不改 `per_device_train_batch_size`
- 不改 `gradient_accumulation_steps`
- 不改 `num_train_epochs`
- 不改 `num_processes`
- 不改每条原始样本对应的 `num_generations`

也就是说，优化目标是：

- 相同的训练步数
- 相同的 batch 组织方式
- 尽量保持相同的 reward / advantage 语义
- 只通过减少 Python / CPU / 分布式通信开销，让 wall-clock 更快

## 2. 当前训练路径背景

### 2.1 训练入口

训练入口在 [trl_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/trl_trainer.py)。

当前流程大致是：

1. 读取 `train/valid/test` 三个 split
2. 根据配置选择：
   - 普通 constrained generation
   - fixed hint constrained generation
   - dynamic hint constrained generation
3. 构建 `GRPOConfig`
4. 进入 `GRPOTrainer` 或它的定制子类

其中：

- fixed hint 会在数据集阶段通过 `train_dataset.map(...)` 注入 `oracle_hint_depth` / `oracle_hint_text`
- dynamic hint 不提前改 train dataset，而是在 runtime 里做 cascade

相关代码：

- [trl_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/trl_trainer.py#L111)
- [trl_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/trl_trainer.py#L126)
- [trl_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/trl_trainer.py#L163)
- [trl_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/trl_trainer.py#L175)

### 2.2 fixed hint 路径

fixed hint 的核心逻辑是：

1. 从离线导出的 `hint_depth_by_index` map 里查每条样本的 hint 深度
2. 从 `reward_model.ground_truth` 中截取前 `hint_depth` 个 SID token，构成 `oracle_hint_text`
3. 在 prompt 后面直接拼接这段 hint 文本
4. 生成后 reward 仍然按“hint + model completion”的完整序列来判定

相关代码：

- [fixed_hint_utils.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_utils.py#L11)
- [fixed_hint_utils.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_utils.py#L15)
- [fixed_hint_utils.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_utils.py#L24)
- [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L45)

### 2.3 dynamic hint 路径

dynamic hint 的核心逻辑在 [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py) 的 `DynamicHintRuleOnlyGRPOTrainer`。

当前实现是：

1. 将 batch 按 `num_generations` 切成多个 group
2. 对所有 unresolved group 先跑 `hint_depth=0`
3. 若某个 group 在该 stage 出现任意一个 `rule hit`，则直接选中这一组输出
4. 未命中的 group 才进入下一层更深的 hint
5. 到 `max_hint_depth` 后，即使仍未 hit，也保留最后一层的输出作为最终输出
6. 最后再统一计算 reward / advantage / metrics

相关代码：

- [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L284)
- [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L474)

### 2.4 当前训练热点的大致分布

从代码结构看，时间主要消耗在以下几段：

1. **constrained generation 本身**
   - 特别是 logits processor 中的前缀约束逻辑
2. **dynamic cascade 的多 stage 生成**
   - 每个 unresolved 子集都要重建 prompt 并再次生成
3. **生成后的 decode + rule reward 判定**
   - dynamic cascade 在 stage 内做一次
   - 最终 reward 计算又做一次
4. **训练日志和跨卡字符串收集**
   - `gather_object(prompts_text/completions_text)`

真正的 GPU 前向当然仍然是大头，但在 constrained beam search 场景下，Python 侧和 CPU 侧的开销也会变得很明显。

## 3. 不改 step 个数的优化原则

这里优先考虑的优化都必须满足：

- 不改变 dataset cardinality
- 不改变 `num_generations`
- 不改变每个 group 的 reward 聚合方式
- 不改变 cascade 的“命中即停、未命中继续”的选择逻辑
- 不通过减少 eval / save 频率来伪装“训练更快”

换句话说，这里只讨论：

- 少做重复工作
- 少做重复 decode / regex / Python 循环
- 少做不必要的跨卡对象收集
- 更高效地组织已有逻辑

## 4. 建议优化点

下面按“收益 / 风险 / 改动量”的综合优先级排序。

### 4.1 优先级最高：优化 constrained logits processor

当前 constrained generation 的 logits processor 明显是高热点。

相关代码：

- [fixed_hint_logit_processor.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_logit_processor.py#L79)
- [logit_processor.py](/Users/fanghaotian/Desktop/src/GenRec/logit_processor.py#L26)

当前问题：

- 每个 decode step 都要按 `batch x beam` 做 Python 双层循环
- 每个 beam 都会 `tolist()`，把 tensor 转成 Python list
- 每个 beam 都会重复做前缀定位和 trie 查询
- 当多个 beam 的当前前缀相同或大量重叠时，没有复用

可以做的优化：

1. 在一次 `__call__` 内部，对 `constraint_prefix_ids` 做 memoization
   - 相同前缀只查一次 allowed tokens
2. 减少不必要的 `tolist()`
   - 尽量只转换需要匹配的后缀段，而不是整条序列
3. 避免每一步从头找 prefix 起点
   - 能增量维护就增量维护
   - 至少也应缩小搜索范围
4. 如果 fixed hint 和普通 constrained 逻辑可以共用更多底层 helper，尽量减少重复实现

预期收益：

- 在 `num_beams=16`、生成长度不短的情况下，这一项最可能带来可见的 wall-clock 改善

风险：

- 这是最容易影响生成语义的部分
- 需要严格做回归验证，确保 allowed tokens 集合完全不变

### 4.2 优先级高：缓存 ground truth token 化结果和格式化后的 base prompt

相关代码：

- [fixed_hint_utils.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_utils.py#L11)
- [fixed_hint_utils.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_utils.py#L15)
- [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L321)

当前问题：

- `build_hint_text(...)` 每次都要重新 `extract_sid_tokens(ground_truth)`
- `build_prompt_with_hint(...)` 每次都重新调用 formatter
- dynamic cascade 每一层 stage 都会重复做这两件事

可以做的优化：

1. 在 example 上预存 `gt_sid_tokens`
2. 在 trainer 内或数据准备阶段预存一次 `base_prompt_text`
3. stage 内只做：
   - `hint_text = ''.join(gt_sid_tokens[:depth])`
   - `prompt_text = base_prompt_text + hint_text`

预期收益：

- 这部分不会带来数量级变化
- 但属于低风险、稳定省 CPU 的优化

风险：

- 很低
- 只要缓存字段不污染下游接口，语义几乎不变

### 4.3 优先级高：复用 dynamic cascade 内已经算过的 rule reward

相关代码：

- stage 内 rule hit 判定：
  [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L344)
- 最终又重新算 reward：
  [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L556)

当前问题：

- dynamic cascade 的每个 stage 已经做了：
  - `batch_decode`
  - `_compute_rule_hit_rewards(...)`
- 但最终被选中的那些 completion，在 `_calculate_rewards(...)` 里又重新算了一遍 rule reward
- 当前 dynamic hint 已经限制为 `reward_mode=rule_only`，因此这部分重复最明显

可以做的优化：

1. 在 `selected_group_outputs` 里一并存下当前 stage 的 `stage_rule_rewards[start:end]`
2. 最终组装 `selected_*` 结果时，把这些 reward 一起收集出来
3. 在 `reward_mode=rule_only` 的 dynamic trainer 路径里，直接构造 `rewards_per_func`
4. 跳过最终的那次 `_calculate_rewards(...)`

预期收益：

- 能减少一次完整的 rule reward 解析流程
- 收益中等，代码也相对干净

风险：

- 需要保证复用的是“最终被选中的那一层”的 reward，而不是中间 stage 的错误版本

### 4.4 优先级中高：减少 train 时的全量字符串 gather/log

相关代码：

- fixed trainer:
  [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L186)
- dynamic trainer:
  [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L602)

当前问题：

- 每个 step 都在 `gather_object(prompts_text)` 和 `gather_object(completions_text)`
- 这会产生额外的：
  - Python object 序列化
  - 跨卡通信
  - 内存占用
- 对训练本身没有直接贡献，主要是日志可观测性开销

可以做的优化：

1. 只在主进程保留部分样本
2. 按 step 采样记录，而不是每步全量记录
3. 或者给字符串日志增加开关，只在 debug / analysis 模式启用

预期收益：

- 对 wall-clock 有一定帮助
- 尤其在多卡下可能比单机更明显

风险：

- 会降低日志细粒度
- 不影响训练语义

### 4.5 优先级中：预解析 ground truth number 序列

相关代码：

- [rewards/ranking_reward.py](/Users/fanghaotian/Desktop/src/GenRec/rewards/ranking_reward.py#L42)
- [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L32)

当前问题：

- `ground_truth` 会被重复正则解析成 number list
- dynamic stage 判定和最终 reward 计算里都会用到

可以做的优化：

1. 在数据准备阶段给样本加一个预解析字段，例如：
   - `ground_truth_numbers`
   - `ground_truth_sid_tokens`
2. reward 计算和 hint 构造都优先使用预解析结果

预期收益：

- 比 logits processor 优化小
- 但和第 4.2 / 4.3 组合起来会更整洁

风险：

- 很低

## 5. 我不建议优先投入的地方

### 5.1 张量拼接和统计部分

例如：

- `pad(...)`
- `torch.cat(...)`
- `view(...).mean(...)`
- `view(...).std(...)`

相关代码：

- [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L80)
- [fixed_hint_grpo_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L490)

原因：

- 这些是标准张量操作
- 相比 constrained beam generation 和 Python 侧约束逻辑，不太可能是主瓶颈

### 5.2 训练前一次性的 dataset.map

相关代码：

- [trl_trainer.py](/Users/fanghaotian/Desktop/src/GenRec/trl_trainer.py#L141)

原因：

- 这只在启动时做一次
- 不影响每个 step 的 steady-state 训练时间

### 5.3 通过改变动态 cascade 语义来“提速”

例如：

- 减少最大 hint depth
- 提前截断某些 group
- 降低 `num_generations`

这些当然会变快，但它们改变的是训练语义，而不是代码实现效率。

本笔记不把这些算作“代码层优化”。

## 6. 推荐实施顺序

如果目标是在较低风险下尽快看到收益，我建议按下面顺序做：

1. **复用 dynamic cascade 已算出的 rule reward**
2. **缓存 `gt_sid_tokens` 和 `base_prompt_text`**
3. **收紧 `gather_object` 字符串日志**
4. **优化 logits processor 的 prefix 查询与缓存**

原因：

- 第 1、2、3 项风险低，比较适合先拿稳定收益
- 第 4 项收益可能最大，但需要更仔细的正确性验证

## 7. 后续实现时的验证重点

每做一项优化，都应检查：

1. 同一输入下，allowed token 集合是否和优化前完全一致
2. fixed hint / dynamic hint 的最终 `rule_reward` 是否一致
3. `dynamic_hint/*` 指标是否保持相同语义
4. 每个 epoch 的 step 个数是否不变
5. 在多卡下日志、gather 和 metrics 是否仍然稳定

尤其是 logits processor 优化，必须把“生成结果完全一致”当作第一优先级，而不是只看速度。

## 8. 总结

当前最值得做的不是微调后处理的张量操作，而是：

- 减少 constrained generation 中的 Python/CPU 热点
- 避免 dynamic cascade 里的重复 prompt/hint 构造
- 避免 rule reward 重复计算
- 减少不必要的字符串 gather/logging

这些优化都可以在 **不改变 step 个数** 的前提下完成，并且大多数不会改变训练语义。

如果后续只做一项，我建议先从“**dynamic cascade 的 reward 复用**”开始；如果要做一项最有潜在收益的，则是“**logits processor 的 prefix 查询优化**”。
