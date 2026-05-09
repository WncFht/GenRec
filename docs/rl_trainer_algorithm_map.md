# GenRec RL Trainer 算法与启动方式整理

## 1. 这份文档在整理什么

这份文档只整理当前 `trl_trainer.py` 这一套 RL 训练入口所覆盖的算法族，以及它们对应的启动方式。

相关代码入口主要有 3 个文件：

- `trl_trainer.py`
- `token_prefix_grpo_trainer.py`
- `fixed_hint_grpo_trainer.py`

可以先把它们理解成一层统一入口加两层算法扩展：

- `trl_trainer.py`：统一参数解析、数据集加载、reward 组装、trainer 选择
- `token_prefix_grpo_trainer.py`：token 级 prefix advantage 这条分支
- `fixed_hint_grpo_trainer.py`：fixed hint / dynamic hint 这条分支

当前系统不是“一个脚本对应一个算法”，而是：

1. 先在 `trl_trainer.py` 里根据参数判断属于哪一类 trainer
2. 再在 trainer 内部决定 reward 和 rollout 的具体行为
3. 最后由不同 shell launcher 去拼这些参数

所以现在看起来乱，本质上是“算法维度”和“启动脚本维度”混在了一起。

## 2. 总体分类

当前这套 RL 训练可以拆成 4 个层次：

1. trainer 类别
2. reward 形式
3. hint 注入方式
4. shell 启动方式

其中最关键的是第 1 层，因为真正决定训练行为的是 trainer 分支。

## 3. Trainer 主分类

### 3.1 原生序列级 GRPO

对应 trainer：

- `trl.GRPOTrainer`

进入条件：

- `fixed_hint_depth_map_path` 没有设置
- `dynamic_hint_max_depth` 没有启用
- `token_level_prefix_advantage=false`

特点：

- 完全走原生 sequence-level GRPO
- reward 是按样本序列给分，不做 token 级 advantage 分配
- 这是最“基础”的分支，也是很多 `rule_only` / `ranking` / `prefix_seq_only` 变体的底座

适用场景：

- plain `rule_only`
- `ranking`
- `ranking_only`
- `prefix_rule_only` 的 sequence-level 版本
- `prefix_only` 的 sequence-level 版本
- `prefix_ranking`

一句话总结：

- 这是“没有 hint、没有 token-level prefix trick”的基线 GRPO。

### 3.2 TokenPrefixGRPOTrainer

对应 trainer：

- `token_prefix_grpo_trainer.py::TokenPrefixGRPOTrainer`

进入条件：

- `fixed_hint_depth_map_path` 没有设置
- `dynamic_hint_max_depth` 没有启用
- `token_level_prefix_advantage=true`

核心想法：

- prefix signal 不再只在序列末端给一个总 reward
- 改成把“prefix 命中了多少 token”直接分配到对应 token 上
- NDCG signal 仍然可以保留，但会以广播或 token-penalty 形式进入 token advantage

这个 trainer 里实际上又有几种子变体：

- `token_adv_total_token_normalize=false`
  - prefix 先做 group-normalize，再把优势只分给命中的 prefix token
- `token_adv_total_token_normalize=true`
  - 先把 prefix token reward 和 ndcg token reward 合成，再做 token-wise group normalize
- `token_level_ndcg_error_token_penalty=true`
  - NDCG 不再广播到所有 completion token，而是只惩罚错误 token

对应的典型算法名：

- `prefix_token`
- `prefix_token_totalnorm`
- `prefix_token_totalnorm_errtok`
- `prefix_token_only`

一句话总结：

- 这是“把 prefix reward 从 sequence-level 改造成 token-level”的一整族方法。

### 3.3 FixedHintRuleOnlyGRPOTrainer

对应 trainer：

- `fixed_hint_grpo_trainer.py::FixedHintRuleOnlyGRPOTrainer`

进入条件：

- `fixed_hint_depth_map_path` 已设置

核心想法：

- 先离线准备一个 `extra_info.index -> oracle_hint_depth` 的 map
- 再把固定深度的 oracle hint 注入到 prompt
- 每个样本只做一次正常 GRPO rollout，不做在线多阶段 cascade

这个分支的关键不是 reward 变了，而是 prompt 被固定 scaffold 了。

额外能力：

- `fixed_hint_depth_cap`
  - 训练时再对 oracle depth 做 cap
- `fixed_hint_unsolved_depth`
  - 离线分析未解样本时使用的 fallback depth
- `fixed_hint_task_names`
  - 只对指定 task 注 hint，其他 task 自动变成 0 hint
- `fixed_hint_apply_to_eval`
  - eval 是否也注入 fixed hint
- `hint_ce_loss_coef`
  - 在 RL loss 之外，加一项 prompt hint token 的 CE 辅助损失

reward 约束：

- 当前只支持 `rule_only`
- 以及 `prefix_rule_only`

一句话总结：

- 这是“离线算好 hint 深度，再固定注入 prompt”的单次生成型 hint RL。

### 3.4 DynamicHintRuleOnlyGRPOTrainer

对应 trainer：

- `fixed_hint_grpo_trainer.py::DynamicHintRuleOnlyGRPOTrainer`

进入条件：

- `dynamic_hint_max_depth > 0`

核心想法：

- hint 深度不是离线固定的
- 训练时从 hint depth `0` 开始在线 cascade
- 每一组 prompt 会逐层尝试更深 hint
- 一旦某层满足命中条件，就选择这一层的 rollout
- 如果一直不中，就走到 `max_hint_depth`

额外能力：

- `dynamic_hint_max_depth`
  - online cascade 的最大 hint 深度
- `dynamic_hint_apply_to_eval`
  - eval 时是否也跑 cascade
- `dynamic_hint_task_names`
  - 只让某些 task 有 dynamic hint budget，其他 task 设成 0
- `hint_ce_loss_coef`
  - 同样支持 hint token CE 辅助损失

reward 约束：

- 当前只支持 `rule_only`
- 以及 `ranking`

一句话总结：

- 这是“训练时在线决定要不要继续加 hint”的 cascade 型 hint RL。

## 4. Reward 形式分类

reward 是另一条独立维度。它和 trainer 类别不是一一对应关系。

当前 `rewards/ranking_reward.py` 里支持：

- `rule_only`
  - 只有 exact hit 才给 1，否则给 0
- `ranking_only`
  - 只有 rank/NDCG 型惩罚
- `prefix_rule_only`
  - 只有 prefix 命中长度 reward
- `prefix_only`
  - prefix reward + ndcg reward
- `prefix_ranking`
  - prefix reward + exact rule reward + ndcg reward
- `ranking`
  - exact rule reward + ndcg reward

可以把它们粗分成三族：

### 4.1 纯 exact-hit 类

- `rule_only`

特点：

- 奖励最稀疏
- 最像标准“答对才给分”

### 4.2 prefix 类

- `prefix_rule_only`
- `prefix_only`
- `prefix_ranking`

特点：

- 奖励更密一些
- 允许“命中前缀也有部分奖励”
- 是 token prefix family 的主要配套 reward

### 4.3 ranking / ndcg 类

- `ranking_only`
- `ranking`
- `prefix_only`
- `prefix_ranking`

特点：

- 会利用同组 beam 内排序信息
- 更依赖 `num_generations=num_beams` 的 group 结构

## 5. 参数组合规则和互斥关系

这部分是后面重构最需要先明确的地方。

### 5.1 trainer 选择优先级

`trl_trainer.py` 当前的 trainer 选择顺序是：

1. `fixed_hint_depth_map_path` 已设置
   - 进入 `FixedHintRuleOnlyGRPOTrainer`
2. 否则如果 `dynamic_hint_max_depth > 0`
   - 进入 `DynamicHintRuleOnlyGRPOTrainer`
3. 否则如果 `token_level_prefix_advantage=true`
   - 进入 `TokenPrefixGRPOTrainer`
4. 否则
   - 回退到原生 `GRPOTrainer`

也就是说：

- fixed hint 优先级最高
- dynamic hint 次之
- token-level prefix 再次
- 普通 GRPO 最后

### 5.2 明确互斥

- `fixed_hint_depth_map_path` 和 `dynamic_hint_max_depth` 不能同时开
- `hint_ce_loss_coef` 只能在 fixed hint 或 dynamic hint 下使用
- dynamic hint 当前不支持 prefix 那一套 reward family
- fixed hint 当前不支持 `ranking`

### 5.3 实际可行组合

可以把当前可行组合理解成下面 4 族：

1. plain GRPO
   - 原生 `GRPOTrainer`
   - reward 可选 `rule_only / ranking / ranking_only / prefix_rule_only / prefix_only / prefix_ranking`
2. token prefix family
   - `TokenPrefixGRPOTrainer`
   - 主要用于 prefix 相关 reward
3. fixed hint family
   - `FixedHintRuleOnlyGRPOTrainer`
   - reward 仅 `rule_only / prefix_rule_only`
4. dynamic hint family
   - `DynamicHintRuleOnlyGRPOTrainer`
   - reward 仅 `rule_only / ranking`

## 6. 启动方式分类

当前启动层大致有 3 层。

### 6.1 最底层：直接调用 `trl_trainer.py`

这是最真实、最完整的启动方式。

基本形态：

```bash
accelerate launch \
  --config_file config/zero2.yaml \
  --num_processes 4 \
  --main_process_port 29516 \
  trl_trainer.py \
  --model ... \
  --data_dir ... \
  --index_path ... \
  --output_dir ... \
  ...
```

优点：

- 能覆盖全部 trainer 分支
- 参数关系最清楚

缺点：

- 参数太多
- 容易拼错

### 6.2 通用封装：`trl_trainer.sh`

这是 repo 根目录下的通用 launcher。

定位：

- 一个偏早期、偏通用的包装器
- 适合 plain GRPO 和基础 reward 配置

它当前显式支持的内容主要是：

- `reward_mode`
- `prefix_reward_normalize`
- `probe_rule_with_zero_weight`
- `token_level_prefix_advantage`

它没有继续暴露的内容包括：

- `fixed_hint_depth_map_path`
- `dynamic_hint_max_depth`
- `hint_ce_loss_coef`
- `token_adv_total_token_normalize`
- `token_level_ndcg_error_token_penalty`

所以它现在已经不是“全功能统一入口”，而更像：

- plain / 早期 prefix family 的通用启动器

这也是当前混乱的重要来源之一：

- 代码能力已经扩展了
- 但顶层通用 launcher 没有同步抽象到位

### 6.3 具体实验 launcher：`hope/.../*.sh`

这批脚本是当前最常用、也最贴近实际实验的启动方式。

它们本质上是：

- 为某个数据集
- 某个 base checkpoint
- 某个算法变体
- 预先填好参数的专用 pipeline 脚本

这层又可以分成两种。

#### 6.3.1 通用 preset 型 launcher

典型代表：

- `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl.sh`

它已经把一部分 prefix family 统一成 preset：

- `prefix_token`
- `prefix_token_totalnorm`
- `prefix_token_totalnorm_errtok`
- `prefix_token_only`
- `prefix_seq_only`
- `rule_only`
- `ranking`
- `ranking_only`

这是目前最接近“算法层抽象” 的脚本层设计。

#### 6.3.2 具体变体直出型 launcher

典型代表：

- `...-rl-rule-only.sh`
- `...-rl-rule-only-fixed-hint.sh`
- `...-rl-rule-only-dynamic-hint.sh`
- `...-rl-rule-only-dynamic-hint-max1.sh`
- `...-rl-ranking-dynamic-hint.sh`
- `...-rl-prefix-token-totalnorm-errtok.sh`

这些脚本的特点是：

- 文件名里直接编码算法名
- 默认参数也直接写死
- 很适合单次复现实验
- 但不利于长期维护

## 7. Games 当前现成启动脚本

结合当前 `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec/`，Games 这边目前明确有 3 条 RL 主线脚本：

### 7.1 plain rule-only

脚本：

- `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only.sh`

默认特点：

- 从 `Games-grec-sft-qwen4B-4-256-dsz0/checkpoint-896` 起跑
- `reward_mode=rule_only`
- `token_level_prefix_advantage=false`
- 不带 hint

对应 trainer：

- 原生 `GRPOTrainer`

### 7.2 fixed hint + rule-only

脚本：

- `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-fixed-hint.sh`

默认流程：

1. 先跑 `analyze_rl_beam_hint.py`
2. 导出 `fixed_hint_depth_map`
3. 再调用 `trl_trainer.py --fixed_hint_depth_map_path ...`

对应 trainer：

- `FixedHintRuleOnlyGRPOTrainer`

这个脚本不是单纯“训练脚本”，而是“分析 + 训练”的两阶段 pipeline。

### 7.3 dynamic hint + rule-only

脚本：

- `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-dynamic-hint.sh`

默认特点：

- `reward_mode=rule_only`
- `dynamic_hint_max_depth=3`
- 在线 cascade hint

对应 trainer：

- `DynamicHintRuleOnlyGRPOTrainer`

## 8. 一张总表

| 算法族 | 触发条件 | 主要参数 | 典型 reward | 典型启动方式 |
|---|---|---|---|---|
| plain GRPO | `fixed_hint_depth_map_path` 关, `dynamic_hint_max_depth` 关, `token_level_prefix_advantage=false` | `reward_mode` | `rule_only`, `ranking`, `prefix_seq_only` | `trl_trainer.py`, `trl_trainer.sh`, `...-rl-rule-only.sh` |
| token prefix GRPO | `token_level_prefix_advantage=true` | `token_adv_total_token_normalize`, `token_level_ndcg_error_token_penalty` | `prefix_only`, `prefix_rule_only` | `trl_trainer.py`, `...-rl.sh --preset ...`, 各类 `prefix-token*.sh` |
| fixed hint GRPO | `fixed_hint_depth_map_path` 开 | `fixed_hint_depth_cap`, `fixed_hint_unsolved_depth`, `hint_ce_loss_coef` | `rule_only`, `prefix_rule_only` | `trl_trainer.py`, `...-rl-rule-only-fixed-hint.sh` |
| dynamic hint GRPO | `dynamic_hint_max_depth>0` | `dynamic_hint_max_depth`, `dynamic_hint_apply_to_eval`, `hint_ce_loss_coef` | `rule_only`, `ranking` | `trl_trainer.py`, `...-rl-rule-only-dynamic-hint.sh`, `...-rl-ranking-dynamic-hint.sh` |

## 9. 当前混乱点总结

如果后面要重构，我认为现在最乱的地方主要有 5 个：

1. trainer 分类和 reward 分类混在一起
   - 例如 `rule_only` 既可能跑 plain GRPO，也可能跑 fixed hint，也可能跑 dynamic hint
2. 同一个算法族同时有“参数组合名”和“脚本文件名”两套命名体系
   - 例如 `prefix_token_totalnorm_errtok`
   - 同时又存在对应 shell 文件名
3. `trl_trainer.py` 才是真统一入口，但 `trl_trainer.sh` 不是全功能统一 launcher
4. fixed hint launcher 是“分析 + 训练”的 pipeline，而 dynamic/plain launcher 是单阶段训练
5. `hope/` 下大量脚本是实验快照，适合复现，不适合继续膨胀成长期接口

## 10. 我建议下一步怎么整理

如果后面继续做重构，我建议先按下面的顺序推进：

1. 先把“trainer 类型”抽象成一级概念
   - `plain`
   - `token_prefix`
   - `fixed_hint`
   - `dynamic_hint`
2. 再把 reward 形式抽象成二级概念
   - `rule_only`
   - `prefix_only`
   - `ranking`
   - 其他组合
3. 最后才把 shell 层统一
   - 通用 launcher 只负责把“trainer 类型 + reward 类型 + dataset preset”翻译成参数
   - 不再让文件名本身承担算法定义

如果按这个方向走，后面最合理的目标应该是：

- `trl_trainer.py` 保持唯一训练入口
- `trl_trainer.sh` 或新的统一 launcher 变成唯一通用启动入口
- `hope/` 下的脚本逐步收敛成 preset 或配置文件，而不是继续新增一批批变体脚本

