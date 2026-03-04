from typing import Any, Optional, Union

import torch
from accelerate.utils import gather
from trl import GRPOTrainer


class TokenPrefixGRPOTrainer(GRPOTrainer):
    """GRPO trainer variant with token-level prefix advantages.

    Design:
    - Prefix signal: token-level (matched prefix tokens only).
    - NDCG signal: sequence-level, broadcast to all valid completion tokens.
    - Rule signal: optional probe only (typically zero-weight in reward_weights).
    """

    def __init__(self, *args, prefix_reward_normalize: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_reward_normalize = prefix_reward_normalize
        self._token_adv_warned = False

    @staticmethod
    def _reward_func_name(reward_func) -> str:
        if hasattr(reward_func, "__name__"):
            return reward_func.__name__
        if hasattr(reward_func, "config") and hasattr(reward_func.config, "_name_or_path"):
            return str(reward_func.config._name_or_path).split("/")[-1]
        return reward_func.__class__.__name__

    @staticmethod
    def _group_normalize(values: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
        grouped = values.view(-1, group_size)
        mean = grouped.mean(dim=1).repeat_interleave(group_size, dim=0)
        std = grouped.std(dim=1).repeat_interleave(group_size, dim=0)
        return (values - mean) / (std + eps)

    def _build_prefix_token_rewards(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        reward_models: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build token rewards and sequence rewards from prefix matches in token space."""
        device = completion_ids.device
        bsz, comp_len = completion_ids.shape
        token_rewards = torch.zeros((bsz, comp_len), dtype=torch.float32, device=device)
        seq_rewards = torch.zeros((bsz,), dtype=torch.float32, device=device)
        eos_id = self.processing_class.eos_token_id

        for i, reward_model in enumerate(reward_models):
            gt_text = reward_model.get("ground_truth", "")
            gt_ids = self.processing_class(gt_text, add_special_tokens=False).input_ids
            if len(gt_ids) == 0:
                continue

            valid_pos = torch.nonzero(completion_mask[i] > 0, as_tuple=False).squeeze(-1)
            if valid_pos.numel() == 0:
                continue
            comp_valid_ids = completion_ids[i, valid_pos]
            # Exclude terminal EOS from prefix matching.
            if comp_valid_ids.numel() > 0 and eos_id is not None and int(comp_valid_ids[-1].item()) == int(eos_id):
                comp_valid_ids = comp_valid_ids[:-1]
                valid_pos = valid_pos[:-1]

            matched = 0
            max_cmp = min(int(comp_valid_ids.numel()), len(gt_ids))
            while matched < max_cmp and int(comp_valid_ids[matched].item()) == int(gt_ids[matched]):
                matched += 1

            if matched == 0:
                continue

            per_tok = 1.0 / float(len(gt_ids)) if self.prefix_reward_normalize else 1.0
            token_rewards[i, valid_pos[:matched]] = per_tok
            seq_rewards[i] = float(matched) * float(per_tok)

        return token_rewards, seq_rewards

    def _get_reward_func_indices(self) -> tuple[Optional[int], Optional[int]]:
        prefix_idx = None
        ndcg_idx = None
        for i, reward_func in enumerate(self.reward_funcs):
            name = self._reward_func_name(reward_func)
            if name == "prefix_rule_reward":
                prefix_idx = i
            elif name == "ndcg_rule_reward":
                ndcg_idx = i
        return prefix_idx, ndcg_idx

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Reuse all generation/logging behavior from upstream first.
        outputs = super()._generate_and_score_completions(inputs)

        prefix_idx, ndcg_idx = self._get_reward_func_indices()
        if prefix_idx is None or ndcg_idx is None:
            # Keep base sequence-level behavior if required reward funcs are missing.
            if not self._token_adv_warned:
                print("[WARN] TokenPrefixGRPOTrainer fallback to sequence-level advantages (missing prefix/ndcg reward).")
                self._token_adv_warned = True
            return outputs

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        completions_text = self.processing_class.batch_decode(outputs["completion_ids"], skip_special_tokens=True)
        reward_kwargs = {key: [example[key] for example in inputs] for key in inputs[0] if key not in ["prompt", "completion"]}
        reward_models = reward_kwargs.get("reward_model", [])

        # 1) Prefix token rewards (local), then local prefix sequence rewards.
        prefix_token_rewards, prefix_seq_local = self._build_prefix_token_rewards(
            outputs["completion_ids"], outputs["completion_mask"], reward_models
        )

        # 2) NDCG sequence rewards (local) from reward function.
        ndcg_func = self.reward_funcs[ndcg_idx]
        ndcg_local = torch.tensor(
            ndcg_func(prompts=prompts, completions=completions_text, **reward_kwargs),
            dtype=torch.float32,
            device=device,
        )

        # 3) Group-normalize both sequence signals globally.
        prefix_seq_global = gather(prefix_seq_local)
        ndcg_global = gather(ndcg_local)
        adv_prefix_global = self._group_normalize(prefix_seq_global, self.num_generations)
        adv_ndcg_global = self._group_normalize(ndcg_global, self.num_generations)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        adv_prefix_local = adv_prefix_global[process_slice]
        adv_ndcg_local = adv_ndcg_global[process_slice]

        # 4) Compose token-level advantages:
        #    A_t = w_prefix * A_prefix * normalized_prefix_token_reward_t
        #        + w_ndcg  * A_ndcg   * completion_mask_t
        prefix_weight = float(self.reward_weights[prefix_idx].item())
        ndcg_weight = float(self.reward_weights[ndcg_idx].item())

        prefix_mass = prefix_token_rewards.sum(dim=1, keepdim=True)
        prefix_dist = torch.where(
            prefix_mass > 0,
            prefix_token_rewards / (prefix_mass + 1e-8),
            torch.zeros_like(prefix_token_rewards),
        )
        completion_mask = outputs["completion_mask"].float()
        token_advantages = (
            prefix_weight * adv_prefix_local.unsqueeze(1) * prefix_dist
            + ndcg_weight * adv_ndcg_local.unsqueeze(1) * completion_mask
        )
        outputs["advantages"] = token_advantages

        mode = "eval" if self.control.should_evaluate else "train"
        nonzero_ratio = ((token_advantages.abs() > 0).float() * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)
        self._metrics[mode]["token_adv_nonzero_ratio"].append(
            self.accelerator.gather_for_metrics(nonzero_ratio).mean().item()
        )
        return outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1).expand_as(per_token_logps)
        elif advantages.dim() == 2:
            if advantages.shape != per_token_logps.shape:
                raise ValueError(
                    f"Token-level advantages shape mismatch: got {tuple(advantages.shape)}, "
                    f"expect {tuple(per_token_logps.shape)}"
                )
        else:
            raise ValueError(f"Unsupported advantages dim={advantages.dim()}")

        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
