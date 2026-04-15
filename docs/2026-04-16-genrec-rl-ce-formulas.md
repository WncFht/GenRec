# GenRec RL 与 Hint-CE 公式说明

- 日期：2026-04-16
- 目的：把当前 `GenRec` 里 `trl` 路线的 RL loss、`hint CE` 辅助项，以及对照 `UFT` 的 `sft_loss` 写成统一公式，便于后续决定 `dapo` 下 CE 应该怎么归一化。

## 1. 符号约定

设：

- `i`：样本 / rollout 序号
- `t`：token 位置
- `T_i`：第 `i` 条 completion 的有效 token 数
- `B`：当前 local batch 的序列数
- `G`：`gradient_accumulation_steps`
- `A_{i,t}`：优势项 advantage
- `\pi_\theta`：当前策略模型
- `\pi_old`：生成这些样本时的旧策略
- `\pi_ref`：reference model
- `m_{i,t}^{resp}`：response token mask
- `m_{i,t}^{hint}`：hint token mask
- `\ell^{ce}_{i,t}`：某个被监督 token 的 next-token cross-entropy

当前 `GenRec` fixed-hint CE 线里，`hint` 指的是 prompt 末尾拼上的 oracle SID prefix，对应的 supervised token 区间由 [build_prompt_hint_shift_mask](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L46) 决定。

## 2. 当前 GenRec 的 KL 与 Entropy 到底怎么进 loss

### 2.1 当前 run 有 KL

当前 `trl` 路线里，`KL` 不是一个独立开关，而是由 `beta` 控制。只要 `beta != 0`，KL 就会被加到每个 completion token 的 loss 上：

```math
\mathrm{KL}_{i,t}
=
\exp\left(\log \pi_{\mathrm{ref}} - \log \pi_\theta\right)
- \left(\log \pi_{\mathrm{ref}} - \log \pi_\theta\right)
- 1
```

```math
\ell^{rl}_{i,t}
=
-\min(r_{i,t} A_{i,t}, \mathrm{clip}(r_{i,t}) A_{i,t})
+ \beta \cdot \mathrm{KL}_{i,t}
```

其中

```math
r_{i,t} = \exp(\log \pi_\theta - \log \pi_{\mathrm{old}})
```

代码位置：

- per-token KL: [grpo_trainer.py#L1681](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1681)
- KL 加进 per-token loss: [grpo_trainer.py#L1727](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1727)

你当前 fixed-hint CE launcher 默认 `BETA=1e-3`，所以这条线当前是带 KL 的。

### 2.2 当前 run 没有 entropy bonus

当前 `GenRec/trl` 路线里，`entropy` 默认不是

```math
- \lambda_{ent} H
```

这种直接加到优化目标里的 regularizer。当前代码里 `entropy` 的角色是：

1. 记录日志指标
2. 如果 `top_entropy_quantile < 1.0`，用 entropy 排序后只保留高熵 token 的 policy-gradient 项

也就是：

```math
\ell^{rl}_{i,t}
\leftarrow
\ell^{rl}_{i,t} \cdot m^{ent}_{i,t}
```

其中 `m^{ent}` 是按熵分位数选出来的 mask。

代码位置：

- 计算 entropy: [grpo_trainer.py#L1661](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1661)
- `top_entropy_quantile` 默认 `1.0`: [grpo_config.py#L215](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_config.py#L215)
- 高熵 mask 生效点: [grpo_trainer.py#L1675](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1675)
- entropy 只做日志: [grpo_trainer.py#L1760](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1760)

所以当前默认情况下：

- `KL` 进 loss
- `entropy` 不直接进 loss

### 2.3 entropy 本身怎么计算

对某个 token 位置的 logits `z`，entropy 是标准 categorical entropy：

```math
H(z)
=
-\sum_v p(v \mid z) \log p(v \mid z)
```

其中

```math
p(v \mid z) = \mathrm{softmax}(z)_v
```

你这边 fixed-hint trainer override 里现在也还是这个公式，见 [_entropy_from_logits](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L68)。

## 3. 当前 GenRec 的四种 RL reduction

在 per-token RL 项 `\ell^{rl}_{i,t}` 已经确定之后，`trl` 还要再选一个 reduction。当前版本支持四种：

### 3.1 `grpo`

先对每条 completion 按长度平均，再对 batch 平均：

```math
L_{\mathrm{grpo}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\frac{\sum_t \ell^{rl}_{i,t} m^{resp}_{i,t}}
{\sum_t m^{resp}_{i,t}}
```

代码： [grpo_trainer.py#L1730](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1730)

### 3.2 `bnpo`

直接按当前 local batch 的所有有效 response token 平均：

```math
L_{\mathrm{bnpo}}
=
\frac{
\sum_i \sum_t \ell^{rl}_{i,t} m^{resp}_{i,t}
}{
\sum_i \sum_t m^{resp}_{i,t}
}
```

代码： [grpo_trainer.py#L1733](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1733)

### 3.3 `dr_grpo`

按固定常数 `B * max_completion_length` 归一化：

```math
L_{\mathrm{dr\_grpo}}
=
\frac{
\sum_i \sum_t \ell^{rl}_{i,t} m^{resp}_{i,t}
}{
B \cdot L_{\max}
}
```

代码： [grpo_trainer.py#L1736](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1736)

### 3.4 `dapo`

按整个 accumulated batch 的有效 response token 总数归一化。设整次 accumulation 跨 rank 汇总后的有效 response token 总数为 `N^{resp}_{global}`，则目标语义是：

```math
L_{\mathrm{dapo}}
=
\frac{
\sum_i \sum_t \ell^{rl}_{i,t} m^{resp}_{i,t}
}{
N^{resp}_{global}
}
```

实现时在每个 rank / micro-step 上用的是：

```math
L^{local}_{\mathrm{dapo}}
=
\frac{
\sum_i \sum_t \ell^{rl}_{i,t} m^{resp}_{i,t}
}{
N^{resp}_{global} / P
}
```

其中 `P` 是 data parallel world size。再通过 DDP 的梯度平均，恢复成全局 token mean 的语义。

代码：

- `num_items_in_batch` 来自全局 completion token 总数: [grpo_trainer.py#L1324](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1324)
- `dapo` reduction: [grpo_trainer.py#L1739](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py#L1739)
- `Trainer` 看到 `num_items_in_batch` 后不会再自动除 `G`: [trainer.py#L4059](/Users/fanghaotian/Desktop/src/GenRec/.venv/lib/python3.12/site-packages/transformers/trainer.py#L4059)

## 4. 当前 GenRec 的 Hint-CE 公式

当前 fixed-hint CE 分支使用 prompt-side logits，对 prompt 末尾被 hint mask 选中的位置做 next-token CE：

```math
\ell^{ce}_{i,t}
=
\mathrm{CE}\bigl(\pi_\theta(x_{i,<t}), x_{i,t}\bigr)
```

只在 `m^{hint}_{i,t}=1` 的位置保留。

当前实现见：

- CE token loss: [fixed_hint_grpo_trainer.py#L145](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L145)
- 当前返回值: [fixed_hint_grpo_trainer.py#L156](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L156)

当前版本的实际公式取决于 `loss_type`。

### 4.1 当 `loss_type != "dapo"` 时

它仍然是 local hint-token mean，再按 `G` 缩放：

```math
L^{current,non\text{-}dapo}_{ce}
=
\frac{
\sum_i \sum_t \ell^{ce}_{i,t} m^{hint}_{i,t}
}{
\sum_i \sum_t m^{hint}_{i,t}
}
\cdot \frac{1}{G}
```

### 4.2 当 `loss_type = "dapo"` 时

当前实现已经改成 strict `dapo` 版本。设整次 accumulation 跨 rank 汇总后的被监督 hint token 总数是 `N^{hint}_{global}`，则：

```math
L^{current,dapo}_{ce}
=
\frac{
\sum_i \sum_t \ell^{ce}_{i,t} m^{hint}_{i,t}
}{
N^{hint}_{global}
}
```

在 local rank / micro-step 上实际 backward 的写法是：

```math
L^{local,current,dapo}_{ce}
=
\frac{
\sum_i \sum_t \ell^{ce}_{i,t} m^{hint}_{i,t}
}{
N^{hint}_{global} / P
}
```

然后总 loss 为：

```math
L_{total}
=
L_{rl}
 \lambda_{ce} L^{current}_{ce}
```

代码： [fixed_hint_grpo_trainer.py#L420](/Users/fanghaotian/Desktop/src/GenRec/fixed_hint_grpo_trainer.py#L420)

这个版本对 `grpo / bnpo / dr_grpo` 是自然的，因为它们本身也会在 loss 内部或外部按 `G` 处理；但对当前实际使用的 `dapo`，它和主 RL loss 不是同一个归一化口径。

## 5. 我们讨论过的三种 CE 版本

### 5.1 版本 A：旧的当前版本

```math
L^{A}_{ce}
=
\mathrm{local\_hint\_token\_mean}
\cdot \frac{1}{G}
```

优点：

- 实现最简单

问题：

- 对 `dapo` 不同口径
- CE 实际有效系数会变成 `\lambda_{ce} / G`

### 5.2 版本 B：最小 dapo 修法

只改掉无条件 `/G`：

```math
L^{B}_{ce}
=
\frac{
\sum_i \sum_t \ell^{ce}_{i,t} m^{hint}_{i,t}
}{
\sum_i \sum_t m^{hint}_{i,t}
}
```

优点：

- 比当前版本少一个明显的缩放错误

问题：

- 仍然是 local hint-token mean
- 不是 strict `dapo`
- 不同 micro-batch 之间是“每个 micro-batch 同权”，不是“每个 hint token 同权”

### 5.3 版本 C：strict dapo CE

设整次 accumulation 跨 rank 汇总后的被监督 hint token 总数是 `N^{hint}_{global}`，则目标语义应为：

```math
L^{C}_{ce}
=
\frac{
\sum_i \sum_t \ell^{ce}_{i,t} m^{hint}_{i,t}
}{
N^{hint}_{global}
}
```

实现时在 local rank / micro-step 上对应：

```math
L^{local,C}_{ce}
=
\frac{
\sum_i \sum_t \ell^{ce}_{i,t} m^{hint}_{i,t}
}{
N^{hint}_{global} / P
}
```

这样经过 DDP 梯度平均后，语义就是全局 hint token mean。

这是和当前 `dapo` 主 RL 项完全同口径的版本，也是 fixed-hint CE trainer 当前在 `loss_type="dapo"` 下采用的实现。

## 6. UFT 对照：它其实不是 dapo

`UFT` 里的 “GRPO” 主要指 advantage 的构造方式，不是 `trl` 这里的 `loss_type='grpo'` reduction。

### 6.1 UFT 的 policy gradient

```math
L^{UFT}_{pg}
=
\mathrm{masked\_mean}\left(
\max(-A \cdot r,\ -A \cdot \mathrm{clip}(r))
, m^{resp}
\right)
```

代码： [core_algos.py#L163](/Users/fanghaotian/Desktop/src/UFT/verl/trainer/ppo/core_algos.py#L163)

### 6.2 UFT 的 hint SFT / CE

```math
L^{UFT}_{sft}
=
\mathrm{masked\_mean}\left(
\ell^{ce}, m^{hint}
\right)
```

代码： [dp_actor.py#L272](/Users/fanghaotian/Desktop/src/UFT/verl/workers/actor/dp_actor.py#L272)

`masked_mean` 本身定义就是：

```math
\mathrm{masked\_mean}(x, m)
=
\frac{\sum x m}{\sum m}
```

代码： [torch_functional.py#L107](/Users/fanghaotian/Desktop/src/UFT/verl/utils/torch_functional.py#L107)

### 6.3 UFT 的总 actor loss

```math
L^{UFT}_{actor}
=
L^{UFT}_{pg}
- \lambda_{ent} L^{UFT}_{ent}
+ \lambda_{sft} L^{UFT}_{sft}
+ \lambda_{kl} L^{UFT}_{kl}
```

最后统一：

```math
L^{UFT}_{backward}
=
\frac{L^{UFT}_{actor}}{G}
```

代码：

- 总 actor loss: [dp_actor.py#L284](/Users/fanghaotian/Desktop/src/UFT/verl/workers/actor/dp_actor.py#L284)
- 最后统一 `/ gradient_accumulation`: [dp_actor.py#L298](/Users/fanghaotian/Desktop/src/UFT/verl/workers/actor/dp_actor.py#L298)

所以 `UFT` 更接近：

- `pg_loss`: local response-token mean
- `sft_loss`: local hint-token mean
- 最后统一 `/ G`

它不是 `dapo`。

## 7. 当前 GenRec 与 UFT 的核心差别

### 7.1 当前 GenRec fixed-hint CE 线

- RL 主项：当前 run 是 `dapo`
- CE 辅助项：当 `loss_type="dapo"` 时，当前实现也是 strict `dapo` CE

所以它是：

```math
\text{dapo RL} + \text{strict-dapo CE}
```

### 7.2 UFT

- RL 主项：local masked mean
- SFT / CE 辅助项：local masked mean
- 最后统一 `/ G`

所以它是：

```math
\text{local-mean RL} + \text{local-mean CE}
```

### 7.3 含义

如果目标是：

1. **保持 `trl` 当前 `dapo` 主项**
   那么更自洽的 CE 版本是 `版本 C：strict dapo CE`

2. **尽量和 UFT 的 mixing semantics 对齐**
   那么更接近的做法不是 `dapo + CE`，而是：
   - 主 RL 也走 local masked mean 风格
   - CE 继续走 local hint-token mean

## 8. 当前最需要记住的一句话

当前 `GenRec` fixed-hint CE 线不是：

```math
\text{UFT-style}
```

也不是：

```math
\text{strict dapo CE}
```

在当前这条 fixed-hint CE launcher 默认配置下，它现在是：

```math
\text{dapo RL} + \text{strict-dapo CE}
```

而 `loss_type != "dapo"` 的分支仍然保留 old local-mean `/ G` 语义。
