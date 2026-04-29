from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import fire
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl.data_utils import maybe_apply_chat_template

from cli_utils import coerce_bool_arg, format_typed_value
from fixed_hint_grpo_trainer import _compute_rule_hit_rewards, build_prompt_hint_shift_mask
from fixed_hint_utils import apply_fixed_hint_depth_to_example, build_hint_text, load_fixed_hint_depth_map
from util import build_fixed_hint_constrained_logits_processor, print_main_process, quiet_non_main_process_logging


try:
    from trl.models import unwrap_model_for_generation
except Exception:  # pragma: no cover - older TRL or lightweight test stubs
    unwrap_model_for_generation = None


def _parse_task_names(raw_task_names) -> Optional[list[str]]:
    if raw_task_names is None:
        return None
    raw_values = raw_task_names if isinstance(raw_task_names, (list, tuple)) else [raw_task_names]
    task_names = []
    for raw_value in raw_values:
        task_names.extend(task_name.strip() for task_name in str(raw_value).split(",") if task_name.strip())
    return task_names or None


def _filter_dataset_by_task_names(dataset, split_name: str, raw_task_names: Optional[str]):
    requested_task_names = _parse_task_names(raw_task_names)
    if not requested_task_names:
        return (
            dataset,
            None,
            sorted(
                {
                    str(example.get("extra_info", {}).get("task", "")).strip()
                    for example in dataset
                    if str(example.get("extra_info", {}).get("task", "")).strip()
                }
            ),
        )

    requested_task_name_set = set(requested_task_names)
    available_task_names = set()
    selected_indices: list[int] = []
    for index, example in enumerate(dataset):
        extra_info = example.get("extra_info")
        if not isinstance(extra_info, dict):
            raise ValueError(f"Missing extra_info.task in {split_name} split at index {index}.")
        task_name = str(extra_info.get("task", "")).strip()
        if not task_name:
            raise ValueError(f"Missing extra_info.task in {split_name} split at index {index}.")
        available_task_names.add(task_name)
        if task_name in requested_task_name_set:
            selected_indices.append(index)

    unknown_task_names = sorted(requested_task_name_set - available_task_names)
    if unknown_task_names:
        raise ValueError(
            f"unknown {split_name} task names: {unknown_task_names}; available tasks: {sorted(available_task_names)}"
        )
    if not selected_indices:
        raise ValueError(
            f"empty filtered {split_name} split after applying tasks {requested_task_names}; "
            f"available tasks: {sorted(available_task_names)}"
        )

    if hasattr(dataset, "select"):
        filtered_dataset = dataset.select(selected_indices)
    else:
        filtered_dataset = dataset.__class__([dataset[index] for index in selected_indices])
    return filtered_dataset, requested_task_names, sorted(available_task_names)


def _build_zero_hint_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        **example,
        "oracle_hint_depth": 0,
        "oracle_hint_text": "",
        "oracle_hint_unsolved": False,
    }


def _default_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def _tokenize_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(input_ids)


def _format_prompt(tokenizer, prompt: Any) -> str:
    return maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"]


def _pad_2d(rows: list[list[int]], padding_value: int, device: torch.device) -> torch.Tensor:
    max_length = max((len(row) for row in rows), default=0)
    if max_length == 0:
        max_length = 1
    padded = [row + [padding_value] * (max_length - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.long, device=device)


def _pad_2d_float(rows: list[list[float]], device: torch.device) -> torch.Tensor:
    max_length = max((len(row) for row in rows), default=0)
    if max_length == 0:
        max_length = 1
    padded = [row + [0.0] * (max_length - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.float32, device=device)


@contextmanager
def _temporary_tokenizer_truncation_side(tokenizer, truncation_side: str):
    old_side = getattr(tokenizer, "truncation_side", None)
    if old_side is None:
        yield
        return
    tokenizer.truncation_side = truncation_side
    try:
        yield
    finally:
        tokenizer.truncation_side = old_side


def build_masked_hint_sft_batch(
    examples: list[dict[str, Any]],
    tokenizer,
    device: torch.device,
    max_seq_length: int = 512,
) -> dict[str, torch.Tensor]:
    pad_token_id = _default_pad_token_id(tokenizer)
    input_id_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    hint_shift_mask_rows: list[list[float]] = []
    hint_token_counts: list[int] = []

    with _temporary_tokenizer_truncation_side(tokenizer, "left"):
        for example in examples:
            prompt_text = _format_prompt(tokenizer, example["prompt"])
            hint_text = str(example.get("oracle_hint_text", ""))
            full_text = f"{prompt_text}{hint_text}"
            hint_ids = _tokenize_text(tokenizer, hint_text) if hint_text else []
            full_ids = _tokenize_text(tokenizer, full_text)
            if max_seq_length > 0 and len(full_ids) > max_seq_length:
                full_ids = full_ids[-max_seq_length:]

            effective_hint_token_count = min(len(hint_ids), max(len(full_ids) - 1, 0))
            shift_mask = build_prompt_hint_shift_mask(
                prompt_lengths=[len(full_ids)],
                hint_token_counts=[effective_hint_token_count],
            )[0]
            input_id_rows.append(full_ids)
            attention_rows.append([1] * len(full_ids))
            hint_shift_mask_rows.append([float(value) for value in shift_mask])
            hint_token_counts.append(effective_hint_token_count)

    return {
        "input_ids": _pad_2d(input_id_rows, padding_value=pad_token_id, device=device),
        "attention_mask": _pad_2d(attention_rows, padding_value=0, device=device),
        "hint_shift_mask": _pad_2d_float(hint_shift_mask_rows, device=device),
        "hint_token_count": torch.tensor(hint_token_counts, dtype=torch.float32, device=device),
    }


class RawExampleCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {"examples": features}


class BaseHintTokenSFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        processing_class,
        max_seq_length: int = 512,
        hint_sft_loss_coef: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processing_class = processing_class
        self.max_seq_length = int(max_seq_length)
        self.hint_sft_loss_coef = float(hint_sft_loss_coef)
        self._hint_sft_metrics: dict[str, dict[str, list[float]]] = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }

    def _prepare_hint_examples(self, examples: list[dict[str, Any]], model) -> list[dict[str, Any]]:
        return examples

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("HintTokenSFTTrainer does not support returning outputs.")
        examples = inputs["examples"]
        examples = self._prepare_hint_examples(examples, model)
        device = next(model.parameters()).device
        batch = build_masked_hint_sft_batch(
            examples,
            tokenizer=self.processing_class,
            device=device,
            max_seq_length=self.max_seq_length,
        )

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]
        hint_shift_mask = batch["hint_shift_mask"].float()

        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view_as(shift_labels)
        local_hint_token_count = hint_shift_mask.sum()
        local_loss_sum = (token_losses * hint_shift_mask).sum()

        gathered_hint_token_count = self.accelerator.gather(local_hint_token_count.detach())
        global_hint_token_count = gathered_hint_token_count.sum()
        zero = local_loss_sum.new_zeros(())
        if torch.isclose(global_hint_token_count, zero):
            loss = local_loss_sum * 0.0
            mean_loss_for_log = zero
        else:
            normalizer = global_hint_token_count / max(int(getattr(self.accelerator, "num_processes", 1)), 1)
            loss = local_loss_sum / normalizer.clamp(min=1.0)
            gathered_loss_sum = self.accelerator.gather(local_loss_sum.detach())
            mean_loss_for_log = gathered_loss_sum.sum() / global_hint_token_count.clamp(min=1.0)

        weighted_loss = self.hint_sft_loss_coef * loss
        mode = "train" if model.training else "eval"
        self._hint_sft_metrics[mode]["hint_sft/loss"].append(float(mean_loss_for_log.detach().item()))
        self._hint_sft_metrics[mode]["hint_sft/token_count"].append(float(global_hint_token_count.detach().item()))
        self._hint_sft_metrics[mode]["hint_sft/loss_weighted"].append(float(weighted_loss.detach().item()))
        return weighted_loss

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(values) / len(values) for key, values in self._hint_sft_metrics[mode].items() if len(values) > 0
        }
        if mode == "eval":
            metrics = {f"eval_{key}": value for key, value in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, *args, **kwargs)
        self._hint_sft_metrics[mode].clear()


class FixedHintTokenSFTTrainer(BaseHintTokenSFTTrainer):
    pass


class DynamicHintTokenSFTTrainer(BaseHintTokenSFTTrainer):
    def __init__(
        self,
        *args,
        logits_processor,
        dynamic_hint_max_depth: int = 3,
        max_prompt_length: int = 512,
        max_completion_length: int = 128,
        num_beams: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logits_processor = logits_processor
        self.dynamic_hint_max_depth = int(dynamic_hint_max_depth)
        self.max_prompt_length = int(max_prompt_length)
        self.max_completion_length = int(max_completion_length)
        self.num_beams = int(num_beams)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.do_sample = bool(do_sample)
        if self.dynamic_hint_max_depth < 0:
            raise ValueError("dynamic_hint_max_depth must be >= 0.")

    def _reset_logits_processor_state(self) -> None:
        if self.logits_processor is None:
            return
        for processor in self.logits_processor:
            if hasattr(processor, "reset"):
                processor.reset()
            elif hasattr(processor, "count"):
                processor.count = 0

    def _unwrapped_model_for_generation(self, model):
        if unwrap_model_for_generation is None:
            unwrapped = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
            return nullcontext(unwrapped)
        gather_ds3 = bool(getattr(self.args, "ds3_gather_for_generation", True))
        return unwrap_model_for_generation(model, self.accelerator, gather_deepspeed3_params=gather_ds3)

    def _generate_stage_completions(self, model, prompts_text: list[str]) -> list[list[str]]:
        if not prompts_text:
            return []
        tokenizer = self.processing_class
        old_padding_side = getattr(tokenizer, "padding_side", None)
        if old_padding_side is not None:
            tokenizer.padding_side = "left"
        try:
            prompt_inputs = tokenizer(
                prompts_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=False,
            )
        finally:
            if old_padding_side is not None:
                tokenizer.padding_side = old_padding_side

        prompt_inputs = self._prepare_inputs(prompt_inputs)
        prompt_width = prompt_inputs["input_ids"].shape[1]
        self._reset_logits_processor_state()
        generation_kwargs = {
            **prompt_inputs,
            "logits_processor": self.logits_processor,
            "max_new_tokens": self.max_completion_length,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "pad_token_id": _default_pad_token_id(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad(), self._unwrapped_model_for_generation(model) as unwrapped_model:
            generated = unwrapped_model.generate(**generation_kwargs)
        completion_ids = generated[:, prompt_width:]
        completions_text = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            completions_text[index * self.num_beams : (index + 1) * self.num_beams]
            for index in range(len(prompts_text))
        ]

    def _build_stage_examples(self, examples: list[dict[str, Any]], hint_depth: int) -> list[dict[str, Any]]:
        enriched_examples = []
        for example in examples:
            enriched = dict(example)
            hint_text = build_hint_text(example["reward_model"]["ground_truth"], hint_depth)
            enriched["oracle_hint_depth"] = int(hint_depth)
            enriched["oracle_hint_text"] = hint_text
            enriched["oracle_hint_unsolved"] = False
            enriched_examples.append(enriched)
        return enriched_examples

    def _prepare_hint_examples(self, examples: list[dict[str, Any]], model) -> list[dict[str, Any]]:
        unresolved_indices = list(range(len(examples)))
        selected_by_index: dict[int, dict[str, Any]] = {}
        stage_rule_hit_fracs: dict[int, float] = {}

        for hint_depth in range(self.dynamic_hint_max_depth + 1):
            if not unresolved_indices:
                break
            stage_source_examples = [examples[index] for index in unresolved_indices]
            stage_examples = self._build_stage_examples(stage_source_examples, hint_depth)
            prompts_text = [
                f"{_format_prompt(self.processing_class, example['prompt'])}{example.get('oracle_hint_text', '')}"
                for example in stage_examples
            ]
            grouped_completions = self._generate_stage_completions(model, prompts_text)
            next_unresolved_indices: list[int] = []
            stage_hit_count = 0

            for source_index, stage_example, completions_text in zip(
                unresolved_indices,
                stage_examples,
                grouped_completions,
            ):
                rule_rewards = _compute_rule_hit_rewards(
                    [stage_example for _ in completions_text],
                    completions_text,
                )
                rule_hit_any = any(reward > 0 for reward in rule_rewards)
                if rule_hit_any:
                    stage_hit_count += 1
                if rule_hit_any or hint_depth == self.dynamic_hint_max_depth:
                    selected = dict(stage_example)
                    selected["dynamic_hint_selected_depth"] = hint_depth
                    selected["dynamic_hint_rule_hit_any"] = rule_hit_any
                    selected_by_index[source_index] = selected
                else:
                    next_unresolved_indices.append(source_index)

            denominator = float(len(unresolved_indices)) if unresolved_indices else 1.0
            stage_rule_hit_fracs[hint_depth] = stage_hit_count / denominator
            unresolved_indices = next_unresolved_indices

        selected_examples = [selected_by_index[index] for index in range(len(examples))]
        selected_depths = [int(example["dynamic_hint_selected_depth"]) for example in selected_examples]
        rule_hits = [bool(example["dynamic_hint_rule_hit_any"]) for example in selected_examples]
        mode = "train" if model.training else "eval"
        if selected_depths:
            self._hint_sft_metrics[mode]["dynamic_hint/selected_hint_depth_mean"].append(
                sum(selected_depths) / float(len(selected_depths))
            )
            self._hint_sft_metrics[mode]["dynamic_hint/max_depth_miss_frac"].append(
                sum(
                    1
                    for depth, hit in zip(selected_depths, rule_hits)
                    if depth == self.dynamic_hint_max_depth and not hit
                )
                / float(len(selected_depths))
            )
        for hint_depth, hit_frac in stage_rule_hit_fracs.items():
            self._hint_sft_metrics[mode][f"dynamic_hint/stage_{hint_depth}_rule_hit_frac"].append(hit_frac)
        return selected_examples


def _load_tokenizer_and_model(
    model_path: str,
    add_tokens_path: Optional[str],
    trust_remote_code: bool,
    bf16: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    added_tokens = 0
    if add_tokens_path:
        tokens_path = Path(add_tokens_path)
        if tokens_path.exists():
            with tokens_path.open(encoding="utf-8") as file:
                new_tokens = json.load(file)
            added_tokens = tokenizer.add_tokens(new_tokens)

    dtype = torch.bfloat16 if bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def main(
    mode: str = "fixed",
    model: str = "saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495",
    data_dir: str = "data/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47/rl",
    index_path: str = "data/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47/id2sid.json",
    add_tokens_path: Optional[str] = None,
    output_dir: str = "saves/qwen2.5-3b/full/Instruments-grec-hint-token-sft",
    fixed_hint_depth_map_path: Optional[str] = None,
    fixed_hint_depth_cap: Optional[int] = None,
    fixed_hint_unsolved_depth: int = 3,
    train_task_names: Optional[str] = None,
    eval_task_names: Optional[str] = None,
    num_beams: int = 16,
    sid_levels: int = -1,
    dynamic_hint_max_depth: int = 3,
    max_prompt_length: int = 512,
    max_seq_length: int = 512,
    max_completion_length: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    do_sample: bool = False,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: float = 5.0,
    learning_rate: float = 3.0e-4,
    hint_sft_loss_coef: float = 1.0,
    logging_steps: int = 1,
    eval_step: int = 100,
    eval_strategy: str = "no",
    save_strategy: str = "steps",
    save_steps: float | int = 0.1,
    save_total_limit: int = 10,
    save_only_model: bool = True,
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 0.3,
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    deepspeed: Optional[str] = None,
    report_to: Optional[str] = None,
    run_name: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = "auto",
    trust_remote_code: bool = True,
):
    quiet_non_main_process_logging()
    raw_bool_args = {
        "do_sample": do_sample,
        "save_only_model": save_only_model,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "trust_remote_code": trust_remote_code,
    }
    parsed_bool_args = {name: coerce_bool_arg(value, name) for name, value in raw_bool_args.items()}
    do_sample = parsed_bool_args["do_sample"]
    save_only_model = parsed_bool_args["save_only_model"]
    bf16 = parsed_bool_args["bf16"]
    gradient_checkpointing = parsed_bool_args["gradient_checkpointing"]
    trust_remote_code = parsed_bool_args["trust_remote_code"]
    if isinstance(deepspeed, str):
        deepspeed_value = deepspeed.strip()
        if deepspeed_value.lower() in {"", "none", "false"}:
            deepspeed = None
        elif Path(deepspeed_value).suffix.lower() in {".yaml", ".yml"}:
            raise ValueError(
                "TrainingArguments(deepspeed=...) expects a DeepSpeed JSON config, not an accelerate YAML config. "
                "With the default launchers, keep config/zero2.yaml as the accelerate launch --config_file and do "
                "not pass trainer-side --deepspeed. Only pass a DeepSpeed JSON here if your accelerate config is "
                "set up for a separate deepspeed_config_file path without overlapping DeepSpeed variables."
            )

    mode = mode.strip().lower()
    if mode not in {"fixed", "dynamic"}:
        raise ValueError("mode must be either 'fixed' or 'dynamic'.")
    if mode == "fixed" and not fixed_hint_depth_map_path:
        raise ValueError("fixed mode requires fixed_hint_depth_map_path.")
    if mode == "dynamic" and not index_path:
        raise ValueError("dynamic mode requires index_path for constrained beam rollout.")

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{data_dir}/train.json",
            "valid": f"{data_dir}/valid.json",
            "test": f"{data_dir}/test.json",
        },
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["valid"]
    train_dataset, resolved_train_task_names, train_available_task_names = _filter_dataset_by_task_names(
        train_dataset,
        split_name="train",
        raw_task_names=train_task_names,
    )
    eval_dataset, resolved_eval_task_names, eval_available_task_names = _filter_dataset_by_task_names(
        eval_dataset,
        split_name="eval",
        raw_task_names=eval_task_names,
    )

    print_main_process(
        f"[INFO] train_task_names={resolved_train_task_names}, "
        f"train_available_tasks={train_available_task_names}, train_size={len(train_dataset)}"
    )
    print_main_process(
        f"[INFO] eval_task_names={resolved_eval_task_names}, "
        f"eval_available_tasks={eval_available_task_names}, eval_size={len(eval_dataset)}"
    )

    if mode == "fixed":
        fixed_hint_map = load_fixed_hint_depth_map(fixed_hint_depth_map_path)

        def _inject_hint(example):
            return apply_fixed_hint_depth_to_example(
                example,
                fixed_hint_map,
                cap_depth=fixed_hint_depth_cap,
                unsolved_depth=fixed_hint_unsolved_depth,
            )

        train_dataset = train_dataset.map(_inject_hint, desc="Inject fixed oracle hint metadata into train dataset")
        eval_dataset = eval_dataset.map(
            _build_zero_hint_example, desc="Attach zero-depth hint metadata to eval dataset"
        )
        train_hint_depths = train_dataset["oracle_hint_depth"]
        hint_depth_hist = {depth: train_hint_depths.count(depth) for depth in sorted(set(train_hint_depths))}
        print_main_process(f"[INFO] fixed_hint_depth_map_path={fixed_hint_depth_map_path}")
        print_main_process(
            f"[INFO] fixed_hint_depth_cap={fixed_hint_depth_cap!r}, fixed_hint_unsolved_depth={fixed_hint_unsolved_depth}"
        )
        print_main_process(f"[INFO] train_oracle_hint_depth_hist={hint_depth_hist}")

    tokenizer, model_obj = _load_tokenizer_and_model(
        model_path=model,
        add_tokens_path=add_tokens_path,
        trust_remote_code=trust_remote_code,
        bf16=bf16,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_steps=eval_step,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_only_model=save_only_model,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        deepspeed=deepspeed,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
    )

    common_trainer_kwargs = dict(
        model=model_obj,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RawExampleCollator(),
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
        hint_sft_loss_coef=hint_sft_loss_coef,
    )
    if mode == "fixed":
        trainer = FixedHintTokenSFTTrainer(**common_trainer_kwargs)
    else:
        logits_processor = build_fixed_hint_constrained_logits_processor(
            index_path,
            tokenizer,
            num_beams=num_beams,
            sid_levels=sid_levels,
        )
        trainer = DynamicHintTokenSFTTrainer(
            **common_trainer_kwargs,
            logits_processor=logits_processor,
            dynamic_hint_max_depth=dynamic_hint_max_depth,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

    effective_hint_lr = float(learning_rate) * float(hint_sft_loss_coef)
    print_main_process(
        "[INFO] raw_bool_args="
        + ", ".join(f"{name}={format_typed_value(value)}" for name, value in raw_bool_args.items())
    )
    print_main_process(
        "[INFO] parsed_bool_args=" + ", ".join(f"{name}={value!r}" for name, value in parsed_bool_args.items())
    )
    print_main_process(
        f"[INFO] mode={mode}, num_beams={num_beams}, dynamic_hint_max_depth={dynamic_hint_max_depth}, "
        f"learning_rate={learning_rate}, hint_sft_loss_coef={hint_sft_loss_coef}, "
        f"effective_hint_lr={effective_hint_lr}, max_seq_length={max_seq_length}, "
        f"max_prompt_length={max_prompt_length}, max_completion_length={max_completion_length}"
    )

    resolved_resume = resume_from_checkpoint
    if isinstance(resolved_resume, str):
        lowered = resolved_resume.strip().lower()
        if lowered in {"", "none", "false"}:
            resolved_resume = None
        elif lowered == "auto":
            resolved_resume = get_last_checkpoint(output_dir)
            if resolved_resume is None:
                print_main_process(f"[INFO] No checkpoint found under {output_dir}, start from scratch.")
            else:
                print_main_process(f"[INFO] Auto resume from checkpoint: {resolved_resume}")

    if resolved_resume is not None:
        if not Path(resolved_resume).is_dir():
            raise FileNotFoundError(f"Checkpoint path not found: {resolved_resume}")
        trainer.train(resume_from_checkpoint=resolved_resume)
    else:
        trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
