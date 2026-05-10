import os
from dataclasses import dataclass
from typing import Optional, Union

from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer

from cli_utils import coerce_bool_arg, format_typed_value
from fixed_hint_grpo_trainer import DynamicHintRuleOnlyGRPOTrainer, FixedHintRuleOnlyGRPOTrainer
from fixed_hint_utils import apply_fixed_hint_depth_to_example, load_fixed_hint_depth_map
from MIMIGenRec import get_grpo_config
from token_prefix_grpo_trainer import TokenPrefixGRPOTrainer
from util import (
    build_constrained_logits_processor,
    build_fixed_hint_constrained_logits_processor,
    print_main_process,
)


@dataclass(frozen=True)
class ParsedBoolArgs:
    save_only_model: bool
    do_sample: bool
    prefix_reward_normalize: bool
    probe_rule_with_zero_weight: bool
    token_level_prefix_advantage: bool
    token_adv_total_token_normalize: bool
    token_level_ndcg_error_token_penalty: bool
    fixed_hint_apply_to_eval: bool
    dynamic_hint_apply_to_eval: bool
    eval_on_start: bool
    bf16: bool


@dataclass(frozen=True)
class DatasetSplitSelection:
    dataset: object
    requested_task_names: Optional[list[str]]
    available_task_names: list[str]


@dataclass(frozen=True)
class HintPreparationResult:
    train_dataset: object
    eval_dataset: object


@dataclass(frozen=True)
class TrainerModeConfig:
    normalized_reward_mode: str
    dynamic_hint_enabled: bool
    dynamic_hint_max_depth: Optional[int]


def _parse_task_names(raw_task_names) -> Optional[list[str]]:
    if raw_task_names is None:
        return None

    raw_values = raw_task_names if isinstance(raw_task_names, (list, tuple)) else [raw_task_names]
    task_names = []
    for raw_value in raw_values:
        task_names.extend(task_name.strip() for task_name in str(raw_value).split(",") if task_name.strip())
    return task_names or None


def _build_zero_hint_example(example: dict) -> dict:
    return {
        **example,
        "oracle_hint_depth": 0,
        "oracle_hint_text": "",
        "oracle_hint_unsolved": False,
    }


def _attach_dynamic_hint_max_depth_override(example: dict, max_hint_depth_override: int) -> dict:
    return {
        **example,
        "dynamic_hint_max_depth_override": int(max_hint_depth_override),
    }


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

    if not available_task_names:
        raise ValueError(
            f"empty filtered {split_name} split after applying tasks {requested_task_names}; available tasks: []"
        )

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


def parse_entrypoint_bool_args(
    raw_bool_args: dict[str, object], *, coerce_bool_arg_fn=coerce_bool_arg
) -> ParsedBoolArgs:
    parsed = {name: coerce_bool_arg_fn(value, name) for name, value in raw_bool_args.items()}
    return ParsedBoolArgs(**parsed)


def build_trainer_mode_config(reward_mode: str, dynamic_hint_max_depth: Optional[int]) -> TrainerModeConfig:
    normalized_reward_mode = reward_mode.strip().lower()
    normalized_dynamic_hint_max_depth = None
    dynamic_hint_enabled = False
    if dynamic_hint_max_depth is not None:
        normalized_dynamic_hint_max_depth = int(dynamic_hint_max_depth)
        dynamic_hint_enabled = normalized_dynamic_hint_max_depth > 0
    return TrainerModeConfig(
        normalized_reward_mode=normalized_reward_mode,
        dynamic_hint_enabled=dynamic_hint_enabled,
        dynamic_hint_max_depth=normalized_dynamic_hint_max_depth,
    )


def validate_mode_configuration(
    mode_config: TrainerModeConfig,
    *,
    fixed_hint_depth_map_path: Optional[str],
    hint_ce_loss_coef: float,
):
    if fixed_hint_depth_map_path is not None and mode_config.dynamic_hint_enabled:
        raise ValueError("fixed_hint_depth_map_path and dynamic_hint_max_depth cannot be enabled at the same time.")
    if hint_ce_loss_coef and fixed_hint_depth_map_path is None and not mode_config.dynamic_hint_enabled:
        raise ValueError("hint_ce_loss_coef currently requires fixed_hint_depth_map_path or dynamic_hint_max_depth.")
    if mode_config.dynamic_hint_enabled and mode_config.normalized_reward_mode not in {"rule_only", "ranking"}:
        raise NotImplementedError(
            "Dynamic hint cascade training currently supports reward_mode=rule_only or reward_mode=ranking only."
        )


def load_dataset_splits(
    data_dir: str,
    *,
    train_task_names: Optional[str],
    eval_task_names: Optional[str],
    load_dataset_fn=load_dataset,
) -> tuple[DatasetSplitSelection, DatasetSplitSelection, object]:
    dataset = load_dataset_fn(
        "json",
        data_files={
            "train": f"{data_dir}/train.json",
            "valid": f"{data_dir}/valid.json",
            "test": f"{data_dir}/test.json",
        },
    )
    train_dataset, resolved_train_task_names, train_available_task_names = _filter_dataset_by_task_names(
        dataset["train"],
        split_name="train",
        raw_task_names=train_task_names,
    )
    eval_dataset, resolved_eval_task_names, eval_available_task_names = _filter_dataset_by_task_names(
        dataset["valid"],
        split_name="eval",
        raw_task_names=eval_task_names,
    )
    return (
        DatasetSplitSelection(
            dataset=train_dataset,
            requested_task_names=resolved_train_task_names,
            available_task_names=train_available_task_names,
        ),
        DatasetSplitSelection(
            dataset=eval_dataset,
            requested_task_names=resolved_eval_task_names,
            available_task_names=eval_available_task_names,
        ),
        dataset["test"],
    )


def log_dataset_selection(
    train_split: DatasetSplitSelection, eval_split: DatasetSplitSelection, *, print_main_process_fn=print_main_process
):
    print_main_process_fn(
        f"[INFO] train_task_names={train_split.requested_task_names}, "
        f"train_available_tasks={train_split.available_task_names}, "
        f"train_size={len(train_split.dataset)}"
    )
    print_main_process_fn(
        f"[INFO] eval_task_names={eval_split.requested_task_names}, "
        f"eval_available_tasks={eval_split.available_task_names}, "
        f"eval_size={len(eval_split.dataset)}"
    )


def _prepare_fixed_hint_datasets(
    train_split: DatasetSplitSelection,
    eval_split: DatasetSplitSelection,
    *,
    reward_mode: str,
    fixed_hint_depth_map_path: str,
    fixed_hint_depth_cap: Optional[int],
    fixed_hint_unsolved_depth: int,
    fixed_hint_task_names: Optional[str],
    fixed_hint_apply_to_eval: bool,
    load_fixed_hint_depth_map_fn=load_fixed_hint_depth_map,
    apply_fixed_hint_depth_to_example_fn=apply_fixed_hint_depth_to_example,
    print_main_process_fn=print_main_process,
) -> HintPreparationResult:
    if reward_mode not in {"rule_only", "prefix_rule_only"}:
        raise NotImplementedError(
            "Fixed oracle hint-depth training currently supports reward_mode=rule_only or "
            "reward_mode=prefix_rule_only only."
        )

    fixed_hint_map = load_fixed_hint_depth_map_fn(fixed_hint_depth_map_path)
    resolved_fixed_hint_task_names = _parse_task_names(fixed_hint_task_names)
    if resolved_fixed_hint_task_names is not None:
        unknown_fixed_hint_task_names = sorted(
            set(resolved_fixed_hint_task_names) - set(train_split.available_task_names)
        )
        if unknown_fixed_hint_task_names:
            raise ValueError(
                "unknown fixed-hint task names: "
                f"{unknown_fixed_hint_task_names}; available tasks: {train_split.available_task_names}"
            )
        fixed_hint_task_name_set = set(resolved_fixed_hint_task_names)
    else:
        fixed_hint_task_name_set = None

    def _inject_hint(example):
        if fixed_hint_task_name_set is not None:
            extra_info = example.get("extra_info")
            if not isinstance(extra_info, dict):
                raise ValueError("Missing extra_info.task while applying fixed-hint task filter.")
            task_name = str(extra_info.get("task", "")).strip()
            if not task_name:
                raise ValueError("Missing extra_info.task while applying fixed-hint task filter.")
            if task_name not in fixed_hint_task_name_set:
                return _build_zero_hint_example(example)
        return apply_fixed_hint_depth_to_example_fn(
            example,
            fixed_hint_map,
            cap_depth=fixed_hint_depth_cap,
            unsolved_depth=fixed_hint_unsolved_depth,
        )

    train_dataset = train_split.dataset.map(_inject_hint, desc="Inject fixed oracle hints into train dataset")
    if fixed_hint_apply_to_eval:
        eval_dataset = eval_split.dataset.map(_inject_hint, desc="Inject fixed oracle hints into eval dataset")
    else:
        eval_dataset = eval_split.dataset.map(
            _build_zero_hint_example, desc="Attach zero-depth fixed hint metadata to eval dataset"
        )

    train_hint_depths = train_dataset["oracle_hint_depth"]
    hint_depth_hist = {depth: train_hint_depths.count(depth) for depth in sorted(set(train_hint_depths))}
    print_main_process_fn("[INFO] fixed_hint_generation_mode=mixed_single_generate")
    print_main_process_fn(f"[INFO] fixed_hint_depth_map_path={fixed_hint_depth_map_path}")
    print_main_process_fn(
        f"[INFO] fixed_hint_depth_cap={fixed_hint_depth_cap!r}, fixed_hint_unsolved_depth={fixed_hint_unsolved_depth}"
    )
    print_main_process_fn(f"[INFO] fixed_hint_task_names={resolved_fixed_hint_task_names}")
    print_main_process_fn(f"[INFO] train_oracle_hint_depth_hist={hint_depth_hist}")
    return HintPreparationResult(train_dataset=train_dataset, eval_dataset=eval_dataset)


def _prepare_dynamic_hint_datasets(
    train_split: DatasetSplitSelection,
    eval_split: DatasetSplitSelection,
    *,
    dynamic_hint_max_depth: int,
    dynamic_hint_apply_to_eval: bool,
    dynamic_hint_task_names: Optional[str],
    print_main_process_fn=print_main_process,
) -> HintPreparationResult:
    resolved_dynamic_hint_task_names = _parse_task_names(dynamic_hint_task_names)
    train_dataset = train_split.dataset
    eval_dataset = eval_split.dataset
    if resolved_dynamic_hint_task_names is not None:
        unknown_dynamic_hint_task_names = sorted(
            set(resolved_dynamic_hint_task_names) - set(train_split.available_task_names)
        )
        if unknown_dynamic_hint_task_names:
            raise ValueError(
                "unknown dynamic-hint task names: "
                f"{unknown_dynamic_hint_task_names}; available tasks: {train_split.available_task_names}"
            )
        dynamic_hint_task_name_set = set(resolved_dynamic_hint_task_names)

        def _attach_dynamic_hint_metadata(example):
            extra_info = example.get("extra_info")
            if not isinstance(extra_info, dict):
                raise ValueError("Missing extra_info.task while applying dynamic-hint task filter.")
            task_name = str(extra_info.get("task", "")).strip()
            if not task_name:
                raise ValueError("Missing extra_info.task while applying dynamic-hint task filter.")
            if task_name in dynamic_hint_task_name_set:
                return _attach_dynamic_hint_max_depth_override(example, dynamic_hint_max_depth)
            return _attach_dynamic_hint_max_depth_override(example, 0)

        train_dataset = train_dataset.map(
            _attach_dynamic_hint_metadata,
            desc="Attach dynamic hint task metadata to train dataset",
        )
        eval_dataset = eval_dataset.map(
            _attach_dynamic_hint_metadata,
            desc="Attach dynamic hint task metadata to eval dataset",
        )
        train_dynamic_hint_caps = train_dataset["dynamic_hint_max_depth_override"]
        dynamic_hint_cap_hist = {
            depth: train_dynamic_hint_caps.count(depth) for depth in sorted(set(train_dynamic_hint_caps))
        }
    else:
        dynamic_hint_cap_hist = None
    print_main_process_fn("[INFO] dynamic_hint_generation_mode=cascade")
    print_main_process_fn(f"[INFO] dynamic_hint_max_depth={dynamic_hint_max_depth}")
    print_main_process_fn(f"[INFO] dynamic_hint_apply_to_eval={dynamic_hint_apply_to_eval}")
    print_main_process_fn(f"[INFO] dynamic_hint_task_names={resolved_dynamic_hint_task_names}")
    if dynamic_hint_cap_hist is not None:
        print_main_process_fn(f"[INFO] train_dynamic_hint_max_depth_override_hist={dynamic_hint_cap_hist}")
    return HintPreparationResult(train_dataset=train_dataset, eval_dataset=eval_dataset)


def prepare_hint_datasets(
    train_split: DatasetSplitSelection,
    eval_split: DatasetSplitSelection,
    *,
    mode_config: TrainerModeConfig,
    fixed_hint_depth_map_path: Optional[str],
    fixed_hint_depth_cap: Optional[int],
    fixed_hint_unsolved_depth: int,
    fixed_hint_task_names: Optional[str],
    fixed_hint_apply_to_eval: bool,
    dynamic_hint_apply_to_eval: bool,
    dynamic_hint_task_names: Optional[str],
    load_fixed_hint_depth_map_fn=load_fixed_hint_depth_map,
    apply_fixed_hint_depth_to_example_fn=apply_fixed_hint_depth_to_example,
    print_main_process_fn=print_main_process,
) -> HintPreparationResult:
    if fixed_hint_depth_map_path is not None:
        return _prepare_fixed_hint_datasets(
            train_split,
            eval_split,
            reward_mode=mode_config.normalized_reward_mode,
            fixed_hint_depth_map_path=fixed_hint_depth_map_path,
            fixed_hint_depth_cap=fixed_hint_depth_cap,
            fixed_hint_unsolved_depth=fixed_hint_unsolved_depth,
            fixed_hint_task_names=fixed_hint_task_names,
            fixed_hint_apply_to_eval=fixed_hint_apply_to_eval,
            load_fixed_hint_depth_map_fn=load_fixed_hint_depth_map_fn,
            apply_fixed_hint_depth_to_example_fn=apply_fixed_hint_depth_to_example_fn,
            print_main_process_fn=print_main_process_fn,
        )
    if mode_config.dynamic_hint_enabled:
        return _prepare_dynamic_hint_datasets(
            train_split,
            eval_split,
            dynamic_hint_max_depth=mode_config.dynamic_hint_max_depth,
            dynamic_hint_apply_to_eval=dynamic_hint_apply_to_eval,
            dynamic_hint_task_names=dynamic_hint_task_names,
            print_main_process_fn=print_main_process_fn,
        )
    return HintPreparationResult(train_dataset=train_split.dataset, eval_dataset=eval_split.dataset)


def build_training_args(
    *,
    output_dir: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    learning_rate: float,
    logging_steps: int,
    eval_step: int,
    eval_strategy: str,
    eval_on_start: bool,
    save_strategy: str,
    save_steps: Union[int, float],
    save_total_limit: int,
    save_only_model: bool,
    warmup_ratio: float,
    max_grad_norm: float,
    optim: str,
    lr_scheduler_type: str,
    max_completion_length: int,
    beta: float,
    num_beams: int,
    bf16: bool,
    deepspeed: Optional[str],
    report_to: Optional[str],
    run_name: Optional[str],
    hint_ce_loss_coef: float,
    reward_weights,
    get_grpo_config_fn=get_grpo_config,
):
    grpo_config_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_steps=eval_step,
        eval_strategy=eval_strategy,
        eval_on_start=eval_on_start,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_only_model=save_only_model,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        max_completion_length=max_completion_length,
        beta=beta,
        num_generations=num_beams,
        bf16=bf16,
        deepspeed=deepspeed,
        report_to=report_to,
        run_name=run_name,
    )
    if hint_ce_loss_coef:
        grpo_config_kwargs["gradient_checkpointing"] = True
        grpo_config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if reward_weights is not None:
        grpo_config_kwargs["reward_weights"] = reward_weights
    return get_grpo_config_fn(**grpo_config_kwargs)


def build_logits_processor(
    *,
    index_path: str,
    tokenizer,
    prefix: Optional[str],
    num_beams: int,
    sid_levels: int,
    use_fixed_hint_processor: bool,
    build_constrained_logits_processor_fn=build_constrained_logits_processor,
    build_fixed_hint_constrained_logits_processor_fn=build_fixed_hint_constrained_logits_processor,
):
    if use_fixed_hint_processor:
        return build_fixed_hint_constrained_logits_processor_fn(
            index_path,
            tokenizer,
            prefix=prefix,
            num_beams=num_beams,
            sid_levels=sid_levels,
        )
    return build_constrained_logits_processor_fn(
        index_path,
        tokenizer,
        prefix=prefix,
        num_beams=num_beams,
        sid_levels=sid_levels,
    )


def build_trainer(
    *,
    fixed_hint_depth_map_path: Optional[str],
    mode_config: TrainerModeConfig,
    token_level_prefix_advantage: bool,
    model,
    training_args,
    reward_funcs,
    train_dataset,
    eval_dataset,
    hint_ce_loss_coef: float,
    dynamic_hint_apply_to_eval: bool,
    prefix_reward_normalize: bool,
    token_adv_total_token_normalize: bool,
    token_level_ndcg_error_token_penalty: bool,
    fixed_hint_trainer_cls=FixedHintRuleOnlyGRPOTrainer,
    dynamic_hint_trainer_cls=DynamicHintRuleOnlyGRPOTrainer,
    token_prefix_trainer_cls=TokenPrefixGRPOTrainer,
    base_trainer_cls=GRPOTrainer,
):
    if fixed_hint_depth_map_path is not None:
        return fixed_hint_trainer_cls(
            model=model,
            args=training_args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            hint_ce_loss_coef=hint_ce_loss_coef,
        )
    if mode_config.dynamic_hint_enabled:
        return dynamic_hint_trainer_cls(
            model=model,
            args=training_args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            hint_ce_loss_coef=hint_ce_loss_coef,
            dynamic_hint_max_depth=mode_config.dynamic_hint_max_depth,
            dynamic_hint_apply_to_eval=dynamic_hint_apply_to_eval,
        )
    if token_level_prefix_advantage:
        return token_prefix_trainer_cls(
            model=model,
            args=training_args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prefix_reward_normalize=prefix_reward_normalize,
            token_adv_total_token_normalize=token_adv_total_token_normalize,
            token_level_ndcg_error_token_penalty=token_level_ndcg_error_token_penalty,
        )
    return base_trainer_cls(
        model=model,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def resolve_resume_checkpoint_path(
    resume_from_checkpoint: Optional[str],
    output_dir: str,
    *,
    get_last_checkpoint_fn=get_last_checkpoint,
    print_main_process_fn=print_main_process,
):
    resolved_resume = resume_from_checkpoint
    if isinstance(resolved_resume, str):
        lowered = resolved_resume.strip().lower()
        if lowered in {"", "none", "false"}:
            return None
        if lowered == "auto":
            resolved_resume = get_last_checkpoint_fn(output_dir)
            if resolved_resume is None:
                print_main_process_fn(f"[INFO] No checkpoint found under {output_dir}, start from scratch.")
            else:
                print_main_process_fn(f"[INFO] Auto resume from checkpoint: {resolved_resume}")
    if resolved_resume is not None and not os.path.isdir(resolved_resume):
        raise FileNotFoundError(f"Checkpoint path not found: {resolved_resume}")
    return resolved_resume


def log_runtime_configuration(
    *,
    raw_bool_args: dict[str, object],
    parsed_bool_args: ParsedBoolArgs,
    reward_mode: str,
    fixed_hint_depth_map_path: Optional[str],
    fixed_hint_depth_cap: Optional[int],
    fixed_hint_unsolved_depth: int,
    hint_ce_loss_coef: float,
    mode_config: TrainerModeConfig,
    save_only_model: bool,
    reward_funcs,
    reward_weights,
    format_typed_value_fn=format_typed_value,
    print_main_process_fn=print_main_process,
):
    print_main_process_fn(
        "[INFO] raw_bool_args="
        + ", ".join(f"{name}={format_typed_value_fn(value)}" for name, value in raw_bool_args.items())
    )
    print_main_process_fn(
        "[INFO] parsed_bool_args="
        + ", ".join(f"{field}={getattr(parsed_bool_args, field)!r}" for field in parsed_bool_args.__dataclass_fields__)
    )
    print_main_process_fn(
        f"[INFO] reward_mode={reward_mode}, "
        f"prefix_reward_normalize={parsed_bool_args.prefix_reward_normalize}, "
        f"probe_rule_with_zero_weight={parsed_bool_args.probe_rule_with_zero_weight}, "
        f"token_level_prefix_advantage={parsed_bool_args.token_level_prefix_advantage}, "
        f"token_adv_total_token_normalize={parsed_bool_args.token_adv_total_token_normalize}, "
        f"token_level_ndcg_error_token_penalty={parsed_bool_args.token_level_ndcg_error_token_penalty}, "
        f"fixed_hint_generation_mode={'mixed_single_generate' if fixed_hint_depth_map_path is not None else 'disabled'}, "
        f"fixed_hint_depth_map_path={fixed_hint_depth_map_path}, "
        f"fixed_hint_depth_cap={fixed_hint_depth_cap}, "
        f"fixed_hint_unsolved_depth={fixed_hint_unsolved_depth}, "
        f"fixed_hint_apply_to_eval={parsed_bool_args.fixed_hint_apply_to_eval}, "
        f"hint_ce_loss_coef={hint_ce_loss_coef}, "
        f"dynamic_hint_generation_mode={'cascade' if mode_config.dynamic_hint_enabled else 'disabled'}, "
        f"dynamic_hint_max_depth={mode_config.dynamic_hint_max_depth}, "
        f"dynamic_hint_apply_to_eval={parsed_bool_args.dynamic_hint_apply_to_eval}, "
        f"save_only_model={save_only_model}, "
        f"num_reward_funcs={len(reward_funcs)}, "
        f"reward_weights={reward_weights}"
    )
