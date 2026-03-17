from __future__ import annotations

import re
from typing import Any, Union

import torch
from accelerate.utils import gather, gather_object
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.grpo_trainer import nanstd
from trl.trainer.utils import pad

from fixed_hint_utils import build_hint_text, build_prompt_with_hint


def _extract_images_from_inputs(inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> list[Any] | None:
    if "images" in inputs[0]:
        images = [example.get("images") for example in inputs]
    elif "image" in inputs[0]:
        images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
    else:
        images = None
    if images is not None and all(img_list == [] for img_list in images):
        images = None
    return images


def _extract_completion_numbers(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"_(\d+)", text)]


def _compute_rule_hit_rewards(
    examples: list[dict[str, Union[torch.Tensor, Any]]],
    completions_text: list[str],
) -> list[float]:
    rewards = []
    for example, completion_text in zip(examples, completions_text):
        full_completion = f"{example.get('oracle_hint_text', '')}{completion_text}"
        gt_numbers = _extract_completion_numbers(example["reward_model"]["ground_truth"])
        completion_numbers = _extract_completion_numbers(full_completion)
        rewards.append(1.0 if completion_numbers == gt_numbers else 0.0)
    return rewards


class FixedHintRuleOnlyGRPOTrainer(GRPOTrainer):
    def _build_hinted_prompts(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> list[str]:
        return [
            build_prompt_with_hint(
                example,
                formatter=lambda prompt: maybe_apply_chat_template({"prompt": prompt}, self.processing_class)[
                    "prompt"
                ],
            )
            for example in inputs
        ]

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        images = _extract_images_from_inputs(inputs)

        prompts: list[Any] = [example["prompt"] for example in inputs]
        prompts_text = self._build_hinted_prompts(inputs)
        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(prompts_text, images)

        if forward_kwargs:
            raise NotImplementedError("Fixed hint trainer currently supports text-only generation batches.")
        if sampling_per_token_logps_list is not None:
            raise NotImplementedError("Fixed hint trainer does not support sampling log-prob correction yet.")

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=None,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=None,
                        **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=None,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }


class DynamicHintRuleOnlyGRPOTrainer(FixedHintRuleOnlyGRPOTrainer):
    def __init__(
        self,
        *args,
        dynamic_hint_max_depth: int = 3,
        dynamic_hint_apply_to_eval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dynamic_hint_max_depth = int(dynamic_hint_max_depth)
        self.dynamic_hint_apply_to_eval = bool(dynamic_hint_apply_to_eval)
        if self.dynamic_hint_max_depth < 0:
            raise ValueError("dynamic_hint_max_depth must be >= 0.")

    def _build_runtime_hinted_example(
        self,
        example: dict[str, Union[torch.Tensor, Any]],
        hint_depth: int,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        enriched = dict(example)
        hint_text = build_hint_text(example["reward_model"]["ground_truth"], hint_depth)
        enriched["oracle_hint_depth"] = int(hint_depth)
        enriched["oracle_hint_text"] = hint_text
        enriched["oracle_hint_unsolved"] = False
        return enriched

    def _generate_dynamic_stage(
        self,
        prompts_text: list[str],
        images: list[Any] | None,
    ) -> tuple[list[Any], list[Any]]:
        (
            prompt_ids_list,
            completion_ids_list,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate_single_turn(prompts_text, images)

        if forward_kwargs:
            raise NotImplementedError("Dynamic hint trainer currently supports text-only generation batches.")
        if sampling_per_token_logps_list is not None:
            raise NotImplementedError("Dynamic hint trainer does not support sampling log-prob correction yet.")

        return prompt_ids_list, completion_ids_list

    def _log_selected_batch_generation_metrics(
        self,
        mode: str,
        prompt_ids_list: list[Any],
        completion_ids_list: list[Any],
    ) -> Any:
        device = self.accelerator.device
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids_list], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()

        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return total_completion_tokens

    def _run_dynamic_hint_cascade(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        images: list[Any] | None,
        max_hint_depth: int,
    ) -> dict[str, Any]:
        if len(inputs) % self.num_generations != 0:
            raise ValueError(
                f"Dynamic hint cascade expects batch size divisible by num_generations={self.num_generations}, "
                f"got len(inputs)={len(inputs)}."
            )

        prompt_groups: list[dict[str, Any]] = []
        for group_index, start in enumerate(range(0, len(inputs), self.num_generations)):
            end = start + self.num_generations
            prompt_groups.append(
                {
                    "group_index": group_index,
                    "inputs": inputs[start:end],
                    "images": None if images is None else images[start:end],
                }
            )

        unresolved_groups = list(prompt_groups)
        selected_group_outputs: dict[int, dict[str, Any]] = {}
        stage_stats: list[dict[str, int]] = []
        is_main_process = bool(getattr(getattr(self, "accelerator", None), "is_main_process", False))
        should_log_stages = is_main_process and max_hint_depth > 0

        for requested_hint_depth in range(max_hint_depth + 1):
            if not unresolved_groups:
                break

            stage_inputs: list[dict[str, Union[torch.Tensor, Any]]] = []
            stage_images: list[Any] | None = [] if images is not None else None
            stage_slices: list[tuple[int, int, dict[str, Any]]] = []

            for group in unresolved_groups:
                group_stage_inputs = [
                    self._build_runtime_hinted_example(example, requested_hint_depth) for example in group["inputs"]
                ]
                start = len(stage_inputs)
                stage_inputs.extend(group_stage_inputs)
                end = len(stage_inputs)
                stage_slices.append((start, end, group))
                if stage_images is not None:
                    stage_images.extend(group["images"])

            stage_prompts_text = self._build_hinted_prompts(stage_inputs)
            if should_log_stages:
                print(
                    "[INFO] dynamic_hint_stage "
                    f"requested_depth={requested_hint_depth} "
                    f"unresolved_groups={len(unresolved_groups)} "
                    f"stage_batch={len(stage_inputs)}"
                )
            stage_prompt_ids_list, stage_completion_ids_list = self._generate_dynamic_stage(
                stage_prompts_text, stage_images
            )

            stage_completions_text = self.processing_class.batch_decode(
                stage_completion_ids_list, skip_special_tokens=True
            )
            stage_rule_rewards = _compute_rule_hit_rewards(
                examples=stage_inputs,
                completions_text=stage_completions_text,
            )

            next_unresolved_groups: list[dict[str, Any]] = []
            stage_rule_hit_group_count = 0
            stage_selected_group_count = 0

            for start, end, group in stage_slices:
                group_rule_hit_any = any(reward > 0 for reward in stage_rule_rewards[start:end])
                if group_rule_hit_any:
                    stage_rule_hit_group_count += 1
                if group_rule_hit_any or requested_hint_depth == max_hint_depth:
                    selected_group_outputs[group["group_index"]] = {
                        "inputs": stage_inputs[start:end],
                        "prompt_texts": stage_prompts_text[start:end],
                        "prompt_ids_list": stage_prompt_ids_list[start:end],
                        "completion_ids_list": stage_completion_ids_list[start:end],
                        "completions_text": stage_completions_text[start:end],
                        "hint_depth": requested_hint_depth,
                        "rule_hit_any": group_rule_hit_any,
                    }
                    stage_selected_group_count += 1
                else:
                    next_unresolved_groups.append(group)

            stage_stats.append(
                {
                    "requested_hint_depth": requested_hint_depth,
                    "evaluated_group_count": len(unresolved_groups),
                    "rule_hit_group_count": stage_rule_hit_group_count,
                    "selected_group_count": stage_selected_group_count,
                    "remaining_group_count": len(next_unresolved_groups),
                }
            )
            if should_log_stages:
                print(
                    "[INFO] dynamic_hint_stage_result "
                    f"requested_depth={requested_hint_depth} "
                    f"rule_hit_groups={stage_rule_hit_group_count} "
                    f"selected_groups={stage_selected_group_count} "
                    f"remaining_groups={len(next_unresolved_groups)}"
                )
            unresolved_groups = next_unresolved_groups

        if len(selected_group_outputs) != len(prompt_groups):
            raise RuntimeError(
                f"Dynamic hint cascade selected {len(selected_group_outputs)} groups for "
                f"{len(prompt_groups)} prompt groups."
            )

        selected_inputs: list[dict[str, Union[torch.Tensor, Any]]] = []
        selected_prompt_texts: list[str] = []
        selected_prompt_ids_list: list[Any] = []
        selected_completion_ids_list: list[Any] = []
        selected_completions_text: list[str] = []
        selected_group_hint_depths: list[int] = []
        selected_group_rule_hits: list[bool] = []

        for group_index in range(len(prompt_groups)):
            group_output = selected_group_outputs[group_index]
            selected_inputs.extend(group_output["inputs"])
            selected_prompt_texts.extend(group_output["prompt_texts"])
            selected_prompt_ids_list.extend(group_output["prompt_ids_list"])
            selected_completion_ids_list.extend(group_output["completion_ids_list"])
            selected_completions_text.extend(group_output["completions_text"])
            selected_group_hint_depths.append(group_output["hint_depth"])
            selected_group_rule_hits.append(group_output["rule_hit_any"])

        local_num_items_in_batch = sum(len(ids) for ids in selected_completion_ids_list)

        return {
            "selected_inputs": selected_inputs,
            "selected_prompt_texts": selected_prompt_texts,
            "selected_prompt_ids_list": selected_prompt_ids_list,
            "selected_completion_ids_list": selected_completion_ids_list,
            "selected_completions_text": selected_completions_text,
            "selected_group_hint_depths": selected_group_hint_depths,
            "selected_group_rule_hits": selected_group_rule_hits,
            "stage_stats": stage_stats,
            "local_num_items_in_batch": local_num_items_in_batch,
        }

    def _log_dynamic_hint_metrics(
        self,
        mode: str,
        stage_stats: list[dict[str, int]],
        selected_group_hint_depths: list[int],
        selected_group_rule_hits: list[bool],
        max_hint_depth: int,
    ) -> None:
        if mode == "eval" and not self.dynamic_hint_apply_to_eval:
            return
        if not selected_group_hint_depths:
            return

        num_groups = len(selected_group_hint_depths)
        self._metrics[mode]["dynamic_hint/selected_hint_depth_mean"].append(
            sum(selected_group_hint_depths) / float(num_groups)
        )
        for hint_depth in range(max_hint_depth + 1):
            selected_count = sum(1 for depth in selected_group_hint_depths if depth == hint_depth)
            self._metrics[mode][f"dynamic_hint/selected_depth_{hint_depth}_frac"].append(
                selected_count / float(num_groups)
            )

        max_depth_miss_count = sum(
            1
            for hint_depth, rule_hit_any in zip(selected_group_hint_depths, selected_group_rule_hits)
            if hint_depth == max_hint_depth and not rule_hit_any
        )
        self._metrics[mode]["dynamic_hint/max_depth_miss_frac"].append(max_depth_miss_count / float(num_groups))

        for stage_stat in stage_stats:
            requested_hint_depth = stage_stat["requested_hint_depth"]
            evaluated_group_count = stage_stat["evaluated_group_count"]
            denominator = float(evaluated_group_count) if evaluated_group_count else 1.0
            self._metrics[mode][f"dynamic_hint/stage_{requested_hint_depth}_rule_hit_frac"].append(
                stage_stat["rule_hit_group_count"] / denominator
            )
            self._metrics[mode][f"dynamic_hint/stage_{requested_hint_depth}_remaining_frac"].append(
                stage_stat["remaining_group_count"] / denominator
            )

        self._logs.setdefault("dynamic_hint_depth", []).extend(selected_group_hint_depths)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        images = _extract_images_from_inputs(inputs)
        prompts: list[Any] = [example["prompt"] for example in inputs]

        max_hint_depth = self.dynamic_hint_max_depth if (mode == "train" or self.dynamic_hint_apply_to_eval) else 0
        cascade = self._run_dynamic_hint_cascade(inputs, images=images, max_hint_depth=max_hint_depth)

        inputs = cascade["selected_inputs"]
        prompts_text = cascade["selected_prompt_texts"]
        prompt_ids_list = cascade["selected_prompt_ids_list"]
        completion_ids_list = cascade["selected_completion_ids_list"]
        completions_text = cascade["selected_completions_text"]
        num_items_in_batch = self._log_selected_batch_generation_metrics(mode, prompt_ids_list, completion_ids_list)

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=None,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=None,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=None,
                        )
            else:
                ref_per_token_logps = None

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._log_dynamic_hint_metrics(
            mode=mode,
            stage_stats=cascade["stage_stats"],
            selected_group_hint_depths=cascade["selected_group_hint_depths"],
            selected_group_rule_hits=cascade["selected_group_rule_hits"],
            max_hint_depth=max_hint_depth,
        )

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
