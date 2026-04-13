# 2026-04-14 Fixed-Hint CE 启动修复记录

- 记录日期：2026-04-14
- 记录目的：记录 `Instruments-grec` 上 `fixed-hint CE` launcher 与 trainer 修复后的默认行为，以及本次远程启动是否已经进入正常训练。

## 一句话结论

截至 `2026-04-14`，`fixed-hint CE` 这条线已经从“启动即报错”修到“能够正常进入训练并稳定打印 train logs”。从用户提供的最新远程日志看，本次 run 已经满足“启动正常”的标准：

- 不再在 startup / first batch 前抛异常
- `gradient_checkpointing=True` 与 `gradient_checkpointing_kwargs={'use_reentrant': False}` 已按预期生效
- `eval_on_start=False` 已按预期生效
- 已经打印出连续训练 step 的 `loss`、`grad_norm`、`reward`、`hint_ce/loss`

因此当前判断是：

- **启动路径已经正常**
- **后续要关注的是训练收敛质量，而不是 launcher / trainer 启动稳定性**

## 1. 这次修了什么

### 1.1 CE launcher 改成 standalone

目标脚本：

- `hope/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh`

当前已经不再依赖其他 shell 脚本转调，而是自己包含：

- 参数解析
- `nohup` / `detach` / `tail` / `run`
- fixed hint map 导出
- accelerate 训练命令

并且默认保持：

- `NUM_BEAMS=16`
- `BEAM_SIZE=16`
- `HINT_CE_LOSS_COEF=0.001`

### 1.2 CE loss 不再额外二次 `model(...)`

在 `fixed_hint_grpo_trainer.py` 中，`hint CE` 现在优先复用主 loss forward 的 prompt-side logits，而不是再额外做一次带梯度的 `model(...)`。

这次修改的直接目的，是避免：

- `fixed-hint main loss forward`
- `hint CE extra forward`

在同一个 train step 里叠成两条反向路径，进而和 `ZeRO-2` / activation checkpointing 组合发生冲突。

### 1.3 兼容旧版 TRL 的 `_get_per_token_logps_and_entropies`

因为当前远程环境里的 TRL 版本比本地假设更老，父类的 `_get_per_token_logps_and_entropies(...)` 不接受某些新 kwargs，例如：

- `mm_token_type_ids`
- `image_position_ids`

所以在 override 中增加了签名过滤，只把父类真正支持的参数继续往下传。

### 1.4 默认不再开局就 eval

这次把 `fixed-hint CE` launcher 的默认值从：

- `EVAL_ON_START=true`

改成了：

- `EVAL_ON_START=false`

这样启动时不再先花一整轮时间做 eval，再进入 train step。

## 2. 本次远程日志为什么说明“已经算正常”

用户给出的最新远程日志里，可以直接确认以下几点：

### 2.1 配置层正确

日志显示：

- `eval_on_start = False`
- `gradient_checkpointing = True`
- `gradient_checkpointing_kwargs = {'use_reentrant': False}`

这说明：

- launcher 默认值已经按预期更新
- trainer 也已经切到了新的 checkpointing 配置

### 2.2 不再在第一步前崩掉

之前的几次问题分别是：

1. `hint CE` 触发额外 forward，和 `ZeRO-2` 发生重复 gradient reduce 冲突
2. 兼容旧版 TRL 时把不支持的 kwargs 直接传给了父类方法

而这次日志已经越过了：

- fixed hint map 导出
- model shard 加载
- deepspeed optimizer 初始化
- wandb offline 初始化

并且真正进入了 train step。

### 2.3 已经打印出连续 train 指标

最新日志里已经出现至少两条训练日志，例如：

- `loss: 0.0038`
- `grad_norm: 1.94`
- `hint_ce/loss: 3.8014`
- `reward: 0.1572`

以及下一条：

- `loss: 0.0034`
- `grad_norm: 2.0385`
- `hint_ce/loss: 3.3904`
- `reward: 0.2070`

这意味着：

- 反向传播已经实际走通
- 优化器已经在更新
- `hint CE` 这部分 loss 也在正常参与训练

所以从工程角度，这次可以判定为“启动正常”。

## 3. 当前默认行为

当前这条 CE 线默认行为是：

- `reward_mode=rule_only`
- `NUM_BEAMS=16`
- `BEAM_SIZE=16`
- `HINT_CE_LOSS_COEF=0.001`
- `EVAL_ON_START=false`
- `gradient_checkpointing=True`
- `gradient_checkpointing_kwargs={'use_reentrant': False}`

也就是说，现在它的默认设定是：

- **保留 activation checkpointing**
- **默认只跑 beam16**
- **默认不做开局 eval**

## 4. 现在最该关注什么

当前最应该看的不再是“会不会启动崩”，而是：

1. `hint_ce/loss` 是否持续稳定在合理区间，而不是爆炸或塌到异常值
2. `reward` / `reward_std` 是否比无 CE 版本更健康
3. 第一个 `eval_step=100` 到来后，离线指标是否有提升
4. 训练速度和显存是否还能接受

换句话说，当前问题已经从：

- “训练起不来”

转成了：

- “这条 CE 辅助损失值不值得继续保留”

## 5. 建议的下一步

建议下一步按下面顺序观察：

1. 跑到至少第一个 `eval_step=100`
2. 对比：
   - plain fixed-hint
   - fixed-hint CE
3. 核心看：
   - `NDCG@10`
   - `HR@10`
   - `NDCG@50`
   - `HR@50`
4. 如果 CE 版没有明显收益，再回头考虑：
   - 调低 `HINT_CE_LOSS_COEF`
   - 或直接移除 CE 分支

当前阶段，不建议再优先怀疑 launcher 或 trainer 启动路径本身。
