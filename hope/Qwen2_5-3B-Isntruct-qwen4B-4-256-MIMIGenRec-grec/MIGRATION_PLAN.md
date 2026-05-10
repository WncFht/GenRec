# Hope Launcher Migration Plan

## Goal

这个目录下的 `*.sh` 目前同时承担了 4 类职责：

1. 环境与后台运行包装
2. 算法默认值与实验命名
3. pipeline 编排
4. 直接拼接 `trl_trainer.py` / `hint_sft_trainer.py` / `analyze_rl_beam_hint.py` 参数

迁移目标不是把所有 shell 简单“翻译”成新 CLI，而是先把它们按职责分类：

- 哪些只是通用 launcher wrapper
- 哪些其实只是 preset/variant
- 哪些是真正有独立 pipeline 的脚本
- 哪些应该最后再迁

最终希望收敛到：

- `trl_trainer.py` 成为唯一 RL Python 入口
- shell 只保留很薄的运行包装
- 变体差异尽量收进 config / preset，而不是继续膨胀文件名

## Current Status

### 已完成

目前已经完成了第一批 train-only RL wrapper 的主线迁移：

| 状态 | 内容 |
|---|---|
| 已完成 | `trl_trainer.py` 已支持 `--config` / `--set ...` 的 config 驱动入口 |
| 已完成 | 通用 RL wrapper `...-rl.sh` 已瘦身为运行层脚本，只保留少量运行时参数解析 |
| 已完成 | 新增共享 helper `_rl_variant_common.sh`，用于 thin wrapper 复用 |
| 已完成 | `rule_only` / `dynamic_hint_rule` / `dynamic_hint_rule_ce` / `dynamic_hint_ranking` / prefix family 的 config 文件已落地 |
| 已完成 | `...-rl-rule-only.sh`、`...-rl-rule-only-dynamic-hint.sh`、`...-rl-rule-only-dynamic-hint-ce.sh`、`...-rl-ranking-dynamic-hint.sh` 已迁成 thin wrapper |
| 已完成 | `...-rl-prefix*.sh` 与 `...-rl-prefix-seq-only.sh` 已迁成 thin wrapper |

### 当前结果

现在 train-only RL 主线已经基本收敛到下面这套结构：

| 层 | 责任 |
|---|---|
| `trl_trainer.py` | 唯一 RL Python 入口，负责 config 解析与训练调度 |
| `configs/rl/instruments/*.json` | 变体配置/preset |
| `...-rl.sh` | 通用运行 wrapper，负责 `nohup / detach / tail / conda / accelerate` |
| `...-rl-variant.sh` | 很薄的 wrapper，只保留各变体的默认 `config/output/run_name/port/eval_on_start` |
| `_rl_variant_common.sh` | 所有 thin wrapper 的共享辅助逻辑 |

### 当前未完成

还没有迁移的重点是：

| 类型 | 当前情况 |
|---|---|
| dynamic task-filter 变体 | 仍是独立大脚本 |
| fixed-hint pipeline | 仍然是 `analyze -> export -> train` 的 shell 编排 |
| hint-token SFT | 仍未收敛到统一入口 |
| evaluate / project root wrapper | 还没动 |

## Current Contents

### File Inventory

| 类别 | 文件 |
|---|---|
| 通用 RL / 分析 / 评测 / SFT 入口 | `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl.sh`, `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-analyze-rl-beam-hint.sh`, `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-evaluate.sh`, `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-hint-token-sft-fixed.sh`, `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-hint-token-sft-dynamic.sh`, `Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec.sh` |
| RL prefix / sequence 变体 | `...-rl-prefix.sh`, `...-rl-prefix-token.sh`, `...-rl-prefix-token-totalnorm.sh`, `...-rl-prefix-token-totalnorm-errtok.sh`, `...-rl-prefix-token-only.sh`, `...-rl-prefix-seq-only.sh` |
| RL plain rule-only | `...-rl-rule-only.sh` |
| RL dynamic-hint 变体 | `...-rl-rule-only-dynamic-hint.sh`, `...-rl-rule-only-dynamic-hint-ce.sh`, `...-rl-rule-only-dynamic-hint-max1.sh`, `...-rl-rule-only-dynamic-hint-sid-only.sh`, `...-rl-rule-only-dynamic-hint-sid-title-desc.sh`, `...-rl-rule-only-dynamic-hint-sid-hint-only-mixed.sh`, `...-rl-ranking-dynamic-hint.sh` |
| RL fixed-hint 变体 | `...-rl-rule-only-fixed-hint.sh`, `...-rl-rule-only-fixed-hint-ce.sh`, `...-rl-rule-only-fixed-hint-sid-only.sh`, `...-rl-rule-only-fixed-hint-sid-title-desc.sh`, `...-rl-rule-only-fixed-hint-sid-hint-only-mixed.sh`, `...-rl-rule-only-fixed-hint-sid-hint-only-mixed-hint-ce.sh`, `...-rl-prefix-seq-only-fixed-hint-sid-only.sh` |
| 文档 | `RL_PRESETS.md` |

### Structural Observation

| 观察 | 含义 |
|---|---|
| 大多数脚本都有 `nohup / detach / tail / run / dry-run / conda / accelerate launch` | 环境层重复非常严重 |
| `rl.sh` 已经吸收了一部分 prefix family 变体 | prefix/plain RL 可以优先迁 |
| fixed-hint 脚本包含 `ANALYZE_CMD + EXPORT_CMD + TRAIN_CMD` 三阶段 | fixed-hint 迁移不能只做 trainer CLI 替换，必须迁 pipeline |
| dynamic-hint 脚本基本只有 `TRAIN_CMD` | dynamic-hint 是很适合第一批迁移到新 CLI 的 |
| 很多脚本差异只在 reward/hint/task/filter/CE/max1 这类小字段 | 这类差异更适合改成 config/preset，而不是单独脚本 |

## Functional Classification

### A. Thin Wrappers That Should Disappear First

这些脚本基本只是对同一个 RL entrypoint 设不同默认值，最适合第一批迁走：

| 文件 | 当前本质 | 迁移目标 |
|---|---|---|
| `...-rl-prefix.sh` | plain RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-prefix-token.sh` | plain RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-prefix-token-totalnorm.sh` | plain RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-prefix-token-totalnorm-errtok.sh` | plain RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-prefix-token-only.sh` | plain RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-prefix-seq-only.sh` | plain RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-rule-only.sh` | plain RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-ranking-dynamic-hint.sh` | dynamic RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-rule-only-dynamic-hint.sh` | dynamic RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-rule-only-dynamic-hint-ce.sh` | dynamic RL wrapper | 并入 `trl_trainer.py --config ...` |
| `...-rl-rule-only-dynamic-hint-max1.sh` | dynamic RL wrapper | 并入 `trl_trainer.py --config ...` |

这批脚本的共同特点：

| 点 | 说明 |
|---|---|
| 不需要离线分析产物 | 可以直接由新 CLI 接管 |
| 主要差异在少量参数 | 很适合变成 config preset |
| 对后续结构帮助最大 | 能最快验证“单入口 + config/preset”是否成立 |

#### A.1 已完成的 Thin Wrappers

下面这些已经完成迁移：

| 文件 | 当前状态 |
|---|---|
| `...-rl.sh` | 已改成通用运行 wrapper |
| `...-rl-rule-only.sh` | 已改成 thin wrapper |
| `...-rl-rule-only-dynamic-hint.sh` | 已改成 thin wrapper |
| `...-rl-rule-only-dynamic-hint-ce.sh` | 已改成 thin wrapper |
| `...-rl-ranking-dynamic-hint.sh` | 已改成 thin wrapper |
| `...-rl-prefix.sh` | 已改成 thin wrapper |
| `...-rl-prefix-token.sh` | 已改成 thin wrapper |
| `...-rl-prefix-token-totalnorm.sh` | 已改成 thin wrapper |
| `...-rl-prefix-token-totalnorm-errtok.sh` | 已改成 thin wrapper |
| `...-rl-prefix-token-only.sh` | 已改成 thin wrapper |
| `...-rl-prefix-seq-only.sh` | 已改成 thin wrapper |

#### A.2 仍待处理的 Thin Wrappers

虽然还是 train-only，但它们还没有进入统一结构：

| 文件 | 当前差异 |
|---|---|
| `...-rl-rule-only-dynamic-hint-max1.sh` | dynamic shallow-budget 变体，需要独立 config |
| `...-rl-rule-only-dynamic-hint-sid-only.sh` | dynamic + sid-only data variant |
| `...-rl-rule-only-dynamic-hint-sid-title-desc.sh` | dynamic + dual-task filter |
| `...-rl-rule-only-dynamic-hint-sid-hint-only-mixed.sh` | dynamic + task-filter / hint-budget 变体 |

这几条都仍然适合留在 Phase 1.x，而不是直接拖到 fixed-hint 阶段。

### B. Pipelines That Need Dedicated Migration

这些脚本不是简单 wrapper，而是真有独立 pipeline：

| 文件 | 当前结构 | 为什么不能只做参数翻译 |
|---|---|---|
| `...-rl-rule-only-fixed-hint.sh` | `analyze -> export -> train` | 需要 Python pipeline 支持 |
| `...-rl-rule-only-fixed-hint-ce.sh` | `analyze -> export -> train` | 同上，只是多 CE 参数 |
| `...-rl-rule-only-fixed-hint-sid-only.sh` | `analyze -> export -> train` | 还带 task/filter 语义 |
| `...-rl-rule-only-fixed-hint-sid-title-desc.sh` | `analyze -> export -> train` | 同上 |
| `...-rl-rule-only-fixed-hint-sid-hint-only-mixed.sh` | `analyze -> export -> train` | 同上 |
| `...-rl-rule-only-fixed-hint-sid-hint-only-mixed-hint-ce.sh` | `analyze -> export -> train` | 同上 |
| `...-rl-prefix-seq-only-fixed-hint-sid-only.sh` | `analyze -> export -> train` | reward 也不同 |
| `...-hint-token-sft-fixed.sh` | `export -> train` | 也属于 fixed pipeline |
| `...-analyze-rl-beam-hint.sh` | analyze-only | 最适合成为通用独立 CLI |

这一批不建议一开始就逐个 shell 改，而应该先在 Python 侧定义稳定 pipeline：

| pipeline 名称 | 责任 |
|---|---|
| `analyze_hint` | 只做 beam hint analysis |
| `rl_fixed_hint` | `analyze/export/train` 一体化 |
| `hint_sft_fixed` | `export/train` 一体化 |

#### B.1 下一阶段最值得优先处理的 Pipeline

当前最合理的下一批对象是：

| 优先级 | 文件 / 目标 | 原因 |
|---:|---|---|
| 1 | `...-analyze-rl-beam-hint.sh` | 先把 analyze-only 独立出来，给 fixed-hint 铺路 |
| 2 | `...-rl-rule-only-fixed-hint.sh` | fixed-hint 主线，价值最高 |
| 3 | `...-rl-rule-only-fixed-hint-ce.sh` | 跟主线最接近，只多一个 CE 参数 |
| 4 | `...-hint-token-sft-fixed.sh` | 复用 fixed-hint export 逻辑 |

也就是说，接下来不是继续无限处理 wrapper，而是该开始迁 **pipeline 本身**。

### C. Scripts That Should Wait

这些脚本可以晚一点迁，因为它们跟“先统一 RL 启动”这个目标关系没那么直接：

| 文件 | 建议时机 |
|---|---|
| `...-evaluate.sh` | 等 RL / SFT train 入口稳定后再迁 |
| `...-hint-token-sft-dynamic.sh` | 等 RL 配置模型稳定后再跟进 |
| `...-grec.sh` | 等 train/eval/analyze 入口都定型后再决定是否保留 |

## Recommended Migration Order

### Phase 1: Migrate Pure RL Wrappers

先迁最薄的 wrapper，目标是验证：

- `trl_trainer.py --config ...`
- `trl_trainer.py --set ...`
- shell 只做运行时包装

建议第一批：

| 优先级 | 文件 | 理由 |
|---:|---|---|
| 1 | `...-rl.sh` | 已经有 `RL_PRESETS.md`，最适合作为总入口替代对象 |
| 2 | `...-rl-rule-only.sh` | plain RL 基线，最简单 |
| 3 | `...-rl-rule-only-dynamic-hint.sh` | dynamic-hint 主线，结构也简单 |
| 4 | `...-rl-rule-only-dynamic-hint-ce.sh` | 验证 CLI 对 CE 附加参数的覆盖 |
| 5 | `...-rl-ranking-dynamic-hint.sh` | 验证 reward family 差异 |
| 6 | `...-rl-prefix-token*.sh` 与 `...-rl-prefix-seq-only.sh` | 用来验证 prefix preset 的映射是否够完整 |

#### Phase 1 Progress

当前进度：

| 项 | 状态 |
|---|---|
| 通用 `rl.sh` 收敛 | 已完成 |
| plain rule-only wrapper | 已完成 |
| dynamic main wrapper | 已完成 |
| dynamic CE wrapper | 已完成 |
| dynamic ranking wrapper | 已完成 |
| prefix family wrappers | 已完成 |

Phase 1 还剩下的主要工作：

| 项 | 状态 |
|---|---|
| dynamic `max1` wrapper | 未开始 |
| dynamic `sid-only` wrapper | 未开始 |
| dynamic `sid-title-desc` wrapper | 未开始 |
| dynamic `sid-hint-only-mixed` wrapper | 未开始 |

### Phase 2: Migrate Fixed-Hint Pipelines

等第一批完成后，再处理 fixed-hint：

| 优先级 | 文件 / 目标 | 理由 |
|---:|---|---|
| 1 | `...-analyze-rl-beam-hint.sh` | 先把 analyze-only 入口做干净 |
| 2 | `...-rl-rule-only-fixed-hint.sh` | fixed-hint 主线，最值得先抽成统一 pipeline |
| 3 | `...-rl-rule-only-fixed-hint-ce.sh` | 在主线稳定后补 CE 分支 |
| 4 | `...-rl-rule-only-fixed-hint-sid-only.sh` | 抽 task/filter 类变体 |
| 5 | 其余 fixed-hint 变体 | 最后统一成 preset/config |

#### Phase 2 Scope Refinement

固定 hint 阶段不应该先从最多变体的脚本下手，而应该按“共享逻辑最强”的顺序做：

| 顺序 | 目标 |
|---:|---|
| 1 | `analyze_rl_beam_hint.py` 的 CLI / config 统一 |
| 2 | `fixed-hint export` 的 Python 化 |
| 3 | `rule-only fixed-hint` 主线 shell 变 thin wrapper |
| 4 | `rule-only fixed-hint CE` 主线 shell 变 thin wrapper |
| 5 | `sid-only / title-desc / mixed` 这些固定 hint 变体再逐步改成 thin wrapper |

这样 fixed-hint 的真正共享层会先出来，而不是继续复制 shell。

### Phase 3: Migrate Hint SFT and Evaluate

最后再迁：

| 文件 | 原因 |
|---|---|
| `...-hint-token-sft-fixed.sh` | 与 fixed-hint export 流程有关系，适合复用 Phase 2 成果 |
| `...-hint-token-sft-dynamic.sh` | 适合在 RL dynamic 入口稳定后再迁 |
| `...-evaluate.sh` | 不阻塞 train pipeline 统一 |
| `...-grec.sh` | 最后再决定是保留为项目总入口，还是拆成 train/eval wrapper |

#### Phase 3 Notes

这一阶段的关键不是“再迁几个 shell”，而是复用前两阶段沉淀下来的能力：

| 复用来源 | 给谁用 |
|---|---|
| config / preset 模型 | `hint-token-sft-*` |
| fixed-hint export pipeline | `hint-token-sft-fixed.sh` |
| 通用运行 wrapper | `evaluate.sh` / project root wrapper |

## Mapping Strategy

### Future Config/Preset Shape

建议后面把这个目录下的大多数变体收敛成 config 文件，而不是保留大量脚本名编码。

例如：

| 当前脚本 | 未来建议 |
|---|---|
| `...-rl-rule-only.sh` | `configs/rl/instruments/rule_only.yaml` |
| `...-rl-rule-only-dynamic-hint.sh` | `configs/rl/instruments/dynamic_hint_rule.yaml` |
| `...-rl-rule-only-dynamic-hint-ce.sh` | `configs/rl/instruments/dynamic_hint_rule_ce.yaml` |
| `...-rl-rule-only-fixed-hint.sh` | `configs/rl/instruments/fixed_hint_rule.yaml` |
| `...-rl-prefix-token-totalnorm-errtok.sh` | `configs/rl/instruments/prefix_token_totalnorm_errtok.yaml` |

当前已经落地的是 JSON 版本：

| 已落地 config | 覆盖内容 |
|---|---|
| `prefix_token.json` | prefix token 主线 |
| `prefix_token_totalnorm.json` | totalnorm 变体 |
| `prefix_token_totalnorm_errtok.json` | errtok 变体 |
| `prefix_token_only.json` | prefix-rule-only token 变体 |
| `prefix_seq_only.json` | prefix-rule-only sequence 变体 |
| `rule_only.json` | plain rule-only |
| `ranking.json` | plain ranking |
| `ranking_only.json` | pure ranking-only |
| `dynamic_hint_rule.json` | dynamic rule-only |
| `dynamic_hint_rule_ce.json` | dynamic rule-only + CE |
| `dynamic_hint_ranking.json` | dynamic ranking |

### Shell End-State

理想终态：

| 保留 | 删除 / 降级 |
|---|---|
| 极少数通用运行 wrapper | 大量算法名脚本 |
| `run.sh` 这类后台/日志包装 | `rule-only-fixed-hint-xxx.sh` 这类变体脚本 |
| 可选兼容别名 | 重复的 `conda/accelerate/nohup` 模板 |

当前已经接近这个状态的 shell：

| 文件 | 当前状态 |
|---|---|
| `...-rl.sh` | 通用运行 wrapper |
| `...-rl-rule-only.sh` | thin wrapper |
| `...-rl-rule-only-dynamic-hint.sh` | thin wrapper |
| `...-rl-rule-only-dynamic-hint-ce.sh` | thin wrapper |
| `...-rl-ranking-dynamic-hint.sh` | thin wrapper |
| `...-rl-prefix*.sh` | thin wrapper |

## First Concrete Migration Task

如果现在就开始动，建议第一步只做这 3 个：

| 顺序 | 目标 |
|---:|---|
| 1 | 让 `...-rl.sh` 改成基于 `trl_trainer.py --config ...` 的总入口 |
| 2 | 迁 `...-rl-rule-only.sh` |
| 3 | 迁 `...-rl-rule-only-dynamic-hint.sh` |

这样做的好处：

| 好处 | 说明 |
|---|---|
| 风险低 | 都是 train-only，不需要 analyze/export pipeline |
| 覆盖面高 | 同时覆盖 plain RL 和 dynamic RL 主线 |
| 能尽快检验新 CLI | 一旦这几条跑通，后续固定 hint 才值得继续抽 |

这一步现在已经基本完成。

## Next Concrete Tasks

从当前状态往后走，建议直接按下面这个顺序：

### Task 1: Finish Remaining Dynamic Thin Wrappers

还没统一的 dynamic 训练变体：

| 文件 | 建议动作 |
|---|---|
| `...-rl-rule-only-dynamic-hint-max1.sh` | 改成 thin wrapper + 新 config |
| `...-rl-rule-only-dynamic-hint-sid-only.sh` | 改成 thin wrapper + 新 config |
| `...-rl-rule-only-dynamic-hint-sid-title-desc.sh` | 改成 thin wrapper + 新 config |
| `...-rl-rule-only-dynamic-hint-sid-hint-only-mixed.sh` | 改成 thin wrapper + 新 config |

### Task 2: Start Fixed-Hint Pipeline Refactor

| 目标 | 说明 |
|---|---|
| 先统一 `analyze` 入口 | 让 fixed-hint 不再强依赖 shell 复制逻辑 |
| 再抽 `fixed export` | 明确 `analyze/export/train` 三阶段接口 |
| 再迁 `rule-only fixed-hint` 主线 | 把 fixed-hint 主线改成 thin wrapper |

### Task 3: Unify Config Naming

当前 config 已经够用，但后面还需要统一命名和组织：

| 当前问题 | 建议 |
|---|---|
| dynamic task-filter 变体还没有 config | 补 config |
| fixed-hint 变体还没有 config | 补 config |
| 目录里已有 `RL_PRESETS.md`，但 shell 已经越来越不靠 preset 逻辑 | 文档改成“config catalog”更合理 |

## Notes

### Why Not Start With Fixed Hint

虽然 fixed-hint 很重要，但它不适合作为第一刀：

| 原因 | 说明 |
|---|---|
| 它不是单阶段训练 | 需要 analyze/export/train 全链路 |
| 脚本内逻辑最重 | 迁移时更容易把 pipeline 和 config 一起搞乱 |
| 更适合作为第二阶段目标 | 先把 train-only 入口收敛，再处理 pipeline |

### What To Avoid

迁移时建议避免：

| 不建议 | 原因 |
|---|---|
| 继续新增新的变体 shell | 会进一步放大重复 |
| 逐个脚本机械替换 | 会把结构问题原样搬过去 |
| 先动 evaluate / hint_sft | 对“统一 RL 启动”帮助不大 |

### Updated Recommendation

结合当前进度，后面最推荐的路线已经不是“继续瘦 `rl.sh`”，而是：

1. 补齐 **remaining dynamic wrappers**
2. 正式开始 **fixed-hint pipeline Python 化**
3. 最后再处理 `hint-token-sft-*` 和 `evaluate.sh`

也就是说，train-only wrapper 这一层已经基本收敛够了，接下来真正有价值的是把 **pipeline 层** 统一掉。
