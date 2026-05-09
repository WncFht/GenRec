# GenRec RL 算法分类与启动方式整理

## 1. 文档目标

这份文档只整理当前 `trl_trainer.py` 这一套 RL 训练系统，重点回答两个问题：

1. 是否开启 hint，以及 hint 是 `none / fixed / dynamic`
2. reward 选的是 `rule / ndcg / prefix` 中的哪一种组合

相关核心代码：

| 文件 | 作用 |
|---|---|
| `trl_trainer.py` | 统一入口：参数解析、数据集加载、reward 组装、trainer 选择 |
| `token_prefix_grpo_trainer.py` | token-level prefix advantage 变体 |
| `fixed_hint_grpo_trainer.py` | fixed hint / dynamic hint 变体 |
| `rewards/ranking_reward.py` | reward mode 定义 |

当前最容易混乱的点不是“算法很多”，而是：

| 维度 | 说明 |
|---|---|
| hint 维度 | 不带 hint、fixed hint、dynamic hint |
| reward 维度 | rule、prefix、ndcg 及其组合 |
| trainer 维度 | 原生 GRPO、TokenPrefixGRPO、FixedHintGRPO、DynamicHintGRPO |
| launcher 维度 | `trl_trainer.py`、`trl_trainer.sh`、`hope/*.sh` |

所以后面看文档时，建议始终按下面这个顺序理解：

1. 先看 hint 分类
2. 再看 reward 分类
3. 最后看这两者在当前代码里怎么落到具体 trainer

## 2. 一级分类：是否开启 hint

### 2.1 总表

| hint 类别 | 关键参数 | 是否需要离线分析 | trainer | 核心含义 |
|---|---|---:|---|---|
| `none` | 不传 `fixed_hint_depth_map_path`，且 `dynamic_hint_max_depth` 不启用 | 否 | `GRPOTrainer` 或 `TokenPrefixGRPOTrainer` | 不给 prompt 额外 hint，直接 RL |
| `fixed` | `fixed_hint_depth_map_path=...` | 是 | `FixedHintRuleOnlyGRPOTrainer` | 每个样本先离线得到一个固定 hint depth，再注入 prompt |
| `dynamic` | `dynamic_hint_max_depth > 0` | 否 | `DynamicHintRuleOnlyGRPOTrainer` | 训练时在线 cascade，从浅 hint 逐步试到深 hint |

### 2.2 `none`：不启用 hint

| 项 | 当前情况 |
|---|---|
| prompt 是否加 hint | 不加 |
| reward 从哪里来 | 直接看 completion 与 ground truth 的关系 |
| trainer 可能是谁 | `GRPOTrainer` 或 `TokenPrefixGRPOTrainer` |
| 什么时候会走 token-prefix trainer | `token_level_prefix_advantage=true` |
| 典型用途 | plain `rule_only`、plain `ranking`、prefix token family |

这里要注意：

| 条件 | 实际分支 |
|---|---|
| `token_level_prefix_advantage=false` | 走原生 `GRPOTrainer` |
| `token_level_prefix_advantage=true` | 走 `TokenPrefixGRPOTrainer` |

也就是说，`none` 下面其实还有两个 trainer 子分支：

| hint | token-level prefix | trainer |
|---|---:|---|
| `none` | false | `GRPOTrainer` |
| `none` | true | `TokenPrefixGRPOTrainer` |

### 2.3 `fixed`：固定 hint

| 项 | 当前情况 |
|---|---|
| prompt 是否加 hint | 加 |
| hint 深度怎么来 | 先离线分析，再导出 `fixed_hint_depth_map` |
| 是否每个样本固定 | 是 |
| 是否在线试多个深度 | 否 |
| trainer | `FixedHintRuleOnlyGRPOTrainer` |

这条线的真实流程是两阶段：

| 阶段 | 做什么 |
|---|---|
| 阶段 1 | 跑 `analyze_rl_beam_hint.py`，根据 beam cascade 结果导出 `fixed_hint_depth_map` |
| 阶段 2 | 跑 `trl_trainer.py --fixed_hint_depth_map_path ...`，把固定深度 hint 注入训练样本 |

常用附加参数：

| 参数 | 作用 |
|---|---|
| `fixed_hint_depth_cap` | 对离线导出的 hint depth 再做训练时裁剪 |
| `fixed_hint_unsolved_depth` | 未解样本使用的 fallback depth |
| `fixed_hint_task_names` | 只给指定 task 注 hint |
| `fixed_hint_apply_to_eval` | eval 时是否也注入 fixed hint |
| `hint_ce_loss_coef` | 给 prompt hint token 再加一个 CE 辅助损失 |

### 2.4 `dynamic`：在线 dynamic hint

| 项 | 当前情况 |
|---|---|
| prompt 是否加 hint | 加 |
| hint 深度怎么来 | 训练时在线决定 |
| 是否每个样本固定 | 否 |
| 是否在线试多个深度 | 是 |
| trainer | `DynamicHintRuleOnlyGRPOTrainer` |

它的核心不是“提前有一个最优 depth”，而是：

| 流程 | 说明 |
|---|---|
| depth=0 开始 | 先尝试无 hint 或最浅 hint |
| 若当前层还不满足条件 | 继续尝试更深 hint |
| 到达命中层 | 选中该层结果 |
| 到达 `max_hint_depth` 仍未命中 | 强制收下最后一层 |

常用附加参数：

| 参数 | 作用 |
|---|---|
| `dynamic_hint_max_depth` | 最大 cascade 深度 |
| `dynamic_hint_apply_to_eval` | eval 时是否也跑 cascade |
| `dynamic_hint_task_names` | 只给指定 task 保留 dynamic hint budget |
| `hint_ce_loss_coef` | 同样支持 prompt-side hint CE 辅助损失 |

## 3. 二级分类：reward 怎么分

### 3.1 总表

当前 `rewards/ranking_reward.py` 里真正支持的 `reward_mode` 如下：

| reward_mode | rule 信号 | prefix 信号 | ndcg / ranking 信号 | 说明 |
|---|---:|---:|---:|---|
| `rule_only` | 是 | 否 | 否 | 只有 exact hit 才给 1 |
| `ranking_only` | 否 | 否 | 是 | 只有 ranking / ndcg 型信号 |
| `prefix_rule_only` | 否 | 是 | 否 | 只有 prefix match reward |
| `prefix_only` | 否 | 是 | 是 | prefix + ndcg |
| `prefix_ranking` | 是 | 是 | 是 | prefix + exact rule + ndcg |
| `ranking` | 是 | 否 | 是 | exact rule + ndcg |

### 3.2 按 reward 原子信号拆开看

#### rule

| 项 | 当前定义 |
|---|---|
| reward 函数 | `rule_reward` |
| 给分逻辑 | completion 完整命中 ground truth 时给 `1.0`，否则 `0.0` |
| 特点 | 最稀疏，最像标准答对才给分 |
| 典型模式 | `rule_only`、`ranking`、`prefix_ranking` |

#### prefix

| 项 | 当前定义 |
|---|---|
| reward 函数 | `prefix_rule_reward` |
| 给分逻辑 | 按照首次错误前匹配到的 prefix 长度给部分分 |
| 是否可归一化 | 是，受 `prefix_reward_normalize` 控制 |
| 特点 | 比 rule 稠密，更适合 token-level 利用 |
| 典型模式 | `prefix_rule_only`、`prefix_only`、`prefix_ranking` |

#### ndcg / ranking

| 项 | 当前定义 |
|---|---|
| reward 函数 | `ndcg_rule_reward` |
| 给分逻辑 | 按 beam group 内的 rank 位置给负权重或排序型信号 |
| 依赖什么 | 强依赖 `num_generations = num_beams` 的 group 结构 |
| 特点 | 不是单样本奖励，而是组内相对排序奖励 |
| 典型模式 | `ranking_only`、`ranking`、`prefix_only`、`prefix_ranking` |

### 3.3 reward 视角下的分类表

| reward 家族 | 包含哪些 mode | 适合什么直觉 |
|---|---|---|
| pure rule | `rule_only` | 只关心“最终是否完全答对” |
| pure prefix | `prefix_rule_only` | 关心“至少先答对前缀” |
| pure ranking | `ranking_only` | 只关心“同组里谁更接近正确” |
| mixed rule + ranking | `ranking` | 同时要 exact hit 和组内排序 |
| mixed prefix + ranking | `prefix_only` | prefix 信号主导，再辅以 ranking |
| mixed all | `prefix_ranking` | rule、prefix、ranking 三个都要 |

## 4. 当前代码里，hint 和 reward 是怎么组合的

### 4.1 组合矩阵

下面这张表是当前最重要的一张表。

| hint 类别 | 可用 reward | 不可用 reward | 对应 trainer | 备注 |
|---|---|---|---|---|
| `none` + 原生 GRPO | `rule_only`、`ranking_only`、`prefix_rule_only`、`prefix_only`、`prefix_ranking`、`ranking` | 无显式限制 | `GRPOTrainer` | 前提是 `token_level_prefix_advantage=false` |
| `none` + token prefix | 主要是 prefix family，也能带 ndcg | 不适合纯 hint trainer 逻辑 | `TokenPrefixGRPOTrainer` | 前提是 `token_level_prefix_advantage=true` |
| `fixed` | `rule_only`、`prefix_rule_only` | `ranking_only`、`prefix_only`、`prefix_ranking`、`ranking` | `FixedHintRuleOnlyGRPOTrainer` | 代码里显式限制 |
| `dynamic` | `rule_only`、`ranking` | `ranking_only`、`prefix_rule_only`、`prefix_only`、`prefix_ranking` | `DynamicHintRuleOnlyGRPOTrainer` | 代码里显式限制 |

### 4.2 trainer 选择优先级

`trl_trainer.py` 里真正的选择顺序是：

| 优先级 | 条件 | trainer |
|---:|---|---|
| 1 | `fixed_hint_depth_map_path` 已设置 | `FixedHintRuleOnlyGRPOTrainer` |
| 2 | 否则，`dynamic_hint_max_depth > 0` | `DynamicHintRuleOnlyGRPOTrainer` |
| 3 | 否则，`token_level_prefix_advantage=true` | `TokenPrefixGRPOTrainer` |
| 4 | 否则 | 原生 `GRPOTrainer` |

这意味着：

| 现象 | 实际含义 |
|---|---|
| 开了 fixed hint | 一定不会进入 dynamic hint 或 token prefix trainer |
| 开了 dynamic hint | 一定不会进入 token prefix trainer |
| 想走 token-level prefix | 必须保证 fixed / dynamic hint 都没开 |

### 4.3 互斥规则

| 组合 | 当前是否允许 | 原因 |
|---|---:|---|
| `fixed_hint_depth_map_path` + `dynamic_hint_max_depth` | 否 | 两种 hint 机制互斥 |
| `hint_ce_loss_coef` + 非 hint 模式 | 否 | CE 辅助损失目前只在 fixed/dynamic hint 下定义 |
| `dynamic hint` + prefix family reward | 否 | 代码里只支持 `rule_only` 或 `ranking` |
| `fixed hint` + `ranking` | 否 | 代码里只支持 `rule_only` 或 `prefix_rule_only` |

## 5. 现在到底是什么情况

### 5.1 如果只按你说的两个维度看

可以把整个系统先压缩成下面这个二维表：

| hint \\ reward | pure rule | pure prefix | pure ranking | mixed |
|---|---|---|---|---|
| `none` | 能跑 | 能跑 | 能跑 | 能跑 |
| `fixed` | 能跑 | 能跑 | 不能跑 | 基本不能跑 |
| `dynamic` | 能跑 | 不能跑 | 不能单独跑 `ranking_only`，但能跑 `ranking` | 只支持 `ranking` 这一种 mixed |

这张表已经能很好解释现在为什么会乱：

| 乱点 | 解释 |
|---|---|
| 同一个 `reward_mode` 能落到不同算法 | 因为它还要看 hint 是否开启 |
| 同一个 hint 家族能搭配的 reward 不一样 | 因为 trainer 实现能力不对称 |
| 文件名像是在表示算法，但其实只表示某个组合 | 因为 shell 脚本把 hint 和 reward 两个维度写死在文件名里了 |

### 5.2 如果再加上 trainer 维度

当前实际存在的是下面 4 类方法：

| 方法类 | hint | reward | trainer | 当前定位 |
|---|---|---|---|---|
| plain sequence GRPO | none | 任意基础 reward 组合 | `GRPOTrainer` | 最基础底座 |
| token prefix GRPO | none | prefix family 为主 | `TokenPrefixGRPOTrainer` | prefix reward 的 token-level 特化 |
| fixed hint GRPO | fixed | rule / prefix-rule-only | `FixedHintRuleOnlyGRPOTrainer` | 离线 scaffold 路线 |
| dynamic hint GRPO | dynamic | rule / ranking | `DynamicHintRuleOnlyGRPOTrainer` | 在线 cascade 路线 |

### 5.3 当前最真实的系统状态

| 维度 | 当前状态 |
|---|---|
| 统一训练入口 | 已经统一到 `trl_trainer.py` |
| 统一算法抽象 | 还没有完全统一 |
| 统一 launcher | 还没有完全统一 |
| 最清晰的分类方法 | 应该先按 hint，再按 reward |
| 当前脚本命名 | 主要是在编码“hint + reward + 一些实验细节”的具体组合 |

## 6. 启动方式怎么对应到这些分类

### 6.1 启动层级总表

| 层级 | 入口 | 适合什么 |
|---|---|---|
| 最底层 | `trl_trainer.py` | 所有真实能力，参数最全 |
| 中间层 | `trl_trainer.sh` | 早期通用封装，主要覆盖 plain / prefix 基础分支 |
| 实验层 | `hope/*.sh` | 某个具体组合的复现实验脚本 |

### 6.2 `trl_trainer.sh` 当前覆盖情况

| 能显式传的参数 | 不能显式传的参数 |
|---|---|
| `reward_mode` | `fixed_hint_depth_map_path` |
| `prefix_reward_normalize` | `dynamic_hint_max_depth` |
| `probe_rule_with_zero_weight` | `hint_ce_loss_coef` |
| `token_level_prefix_advantage` | `token_adv_total_token_normalize` |
|  | `token_level_ndcg_error_token_penalty` |

所以它现在更像：

| 结论 | 说明 |
|---|---|
| 它不是全功能统一 launcher | 因为 hint family 的关键参数没有暴露全 |
| 它更像基础封装 | 适合 plain GRPO 和早期 prefix family |

### 6.3 `hope/*.sh` 是怎么编码组合的

| 脚本名风格 | 实际编码了什么 |
|---|---|
| `...-rl-rule-only.sh` | `hint=none`，`reward=rule_only` |
| `...-rl-rule-only-fixed-hint.sh` | `hint=fixed`，`reward=rule_only` |
| `...-rl-rule-only-dynamic-hint.sh` | `hint=dynamic`，`reward=rule_only` |
| `...-rl-ranking-dynamic-hint.sh` | `hint=dynamic`，`reward=ranking` |
| `...-rl-prefix-token-totalnorm-errtok.sh` | `hint=none`，`reward=prefix_only`，再加 token-prefix 子变体参数 |

## 7. Games 这边当前明确存在的主线

### 7.1 Games 主线表

| 脚本 | hint | reward | trainer | 额外说明 |
|---|---|---|---|---|
| `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only.sh` | none | `rule_only` | `GRPOTrainer` | 从 `checkpoint-896` 起跑 |
| `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-fixed-hint.sh` | fixed | `rule_only` | `FixedHintRuleOnlyGRPOTrainer` | 先分析再训练 |
| `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-dynamic-hint.sh` | dynamic | `rule_only` | `DynamicHintRuleOnlyGRPOTrainer` | 在线 cascade |

### 7.2 Games fixed hint 的特殊点

| 项 | 当前情况 |
|---|---|
| 是否单阶段训练 | 否 |
| 前置步骤 | 需要 `analyze_rl_beam_hint.py` 导出 fixed hint map |
| 为什么和 dynamic/plain 不一样 | 因为 fixed hint 依赖离线 oracle depth 结果 |

## 8. 我建议后面文档和代码都按这个方式继续整理

### 8.1 文档分类建议

以后描述算法时，建议统一先写成：

| 维度 | 示例 |
|---|---|
| hint | `none / fixed / dynamic` |
| reward | `rule_only / prefix_only / ranking / ...` |
| trainer 子变体 | 如 `token_prefix_totalnorm_errtok` |

比如：

| 旧说法 | 更清晰的说法 |
|---|---|
| `rule_only fixed hint` | `hint=fixed, reward=rule_only` |
| `dynamic ranking` | `hint=dynamic, reward=ranking` |
| `prefix token totalnorm errtok` | `hint=none, reward=prefix_only, token-prefix totalnorm errtok` |

### 8.2 代码抽象建议

如果后面要重构，我建议先把接口层抽成这三个字段：

| 字段 | 值 |
|---|---|
| `hint_mode` | `none / fixed / dynamic` |
| `reward_mode` | 继续沿用当前 `rule_only / prefix_only / ranking / ...` |
| `trainer_variant` | `seq / token_prefix / fixed_hint / dynamic_hint` |

这样：

| 好处 | 说明 |
|---|---|
| 分类更稳定 | 先按 hint，再按 reward，不会混 |
| launcher 更容易统一 | 文件名不必承载全部实验定义 |
| 文档更容易维护 | 表格里直接列合法组合即可 |

