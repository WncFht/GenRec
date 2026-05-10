import argparse
import json
from collections.abc import Callable
from typing import Any, Optional, Union

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer

from cli_utils import coerce_bool_arg, format_typed_value
from fixed_hint_grpo_trainer import DynamicHintRuleOnlyGRPOTrainer, FixedHintRuleOnlyGRPOTrainer
from fixed_hint_utils import apply_fixed_hint_depth_to_example, load_fixed_hint_depth_map
from MIMIGenRec import MIMIGenRec, get_grpo_config
from rewards.ranking_reward import build_reward_setup
from rl_trainer_config import RLTrainerConfig
from token_prefix_grpo_trainer import TokenPrefixGRPOTrainer
from trl_trainer_runtime import (
    build_logits_processor,
    build_trainer,
    build_trainer_mode_config,
    build_training_args,
    load_dataset_splits,
    log_dataset_selection,
    log_runtime_configuration,
    parse_entrypoint_bool_args,
    prepare_hint_datasets,
    resolve_resume_checkpoint_path,
    validate_mode_configuration,
)
from util import (
    build_constrained_logits_processor,
    build_fixed_hint_constrained_logits_processor,
    print_main_process,
    quiet_non_main_process_logging,
)


def run_training_kwargs(
    model: str = "saves/qwen2.5-0.5b/full/Industrial_and_Scientific-sft-dsz0",
    index_path: str = "data/Industrial_and_Scientific/Industrial_and_Scientific.index.json",
    sid_levels: int = -1,
    prefix: Optional[str] = None,
    num_beams: int = 16,
    # num_beams: int = 2,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    data_dir: str = "data/Industrial_and_Scientific/rl",
    output_dir: str = "rl_outputs/qwen2.5-0.5b-instruct-grpo",
    # output_dir: str = "rl_outputs/MiniOneRec-MiniMind2-grpo",
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    num_train_epochs: int = 2,
    learning_rate: float = 1e-5,
    logging_steps: int = 1,
    eval_step: int = 100,
    eval_strategy: str = "steps",
    eval_on_start: bool = False,
    save_strategy: str = "steps",
    save_steps: Union[int, float] = 0.1,
    save_total_limit: int = 3,
    save_only_model: bool = False,
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 0.3,
    optim: str = "paged_adamw_32bit",
    lr_scheduler_type: str = "cosine",
    max_completion_length: int = 128,
    beta: float = 1e-3,
    repetition_penalty: float = 1.0,
    do_sample: bool = False,
    reward_mode: str = "prefix_only",
    prefix_reward_normalize: bool = True,
    probe_rule_with_zero_weight: bool = True,
    token_level_prefix_advantage: bool = True,
    token_adv_total_token_normalize: bool = False,
    token_level_ndcg_error_token_penalty: bool = False,
    fixed_hint_depth_map_path: Optional[str] = None,
    fixed_hint_depth_cap: Optional[int] = None,
    fixed_hint_unsolved_depth: int = 3,
    fixed_hint_task_names: Optional[str] = None,
    fixed_hint_apply_to_eval: bool = False,
    hint_ce_loss_coef: float = 0.0,
    dynamic_hint_max_depth: Optional[int] = None,
    dynamic_hint_apply_to_eval: bool = False,
    dynamic_hint_task_names: Optional[str] = None,
    train_task_names: Optional[str] = None,
    eval_task_names: Optional[str] = None,
    bf16: bool = True,
    deepspeed: Optional[str] = None,
    report_to: Optional[str] = None,
    run_name: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = "auto",
):
    quiet_non_main_process_logging()

    raw_bool_args = {
        "save_only_model": save_only_model,
        "do_sample": do_sample,
        "prefix_reward_normalize": prefix_reward_normalize,
        "probe_rule_with_zero_weight": probe_rule_with_zero_weight,
        "token_level_prefix_advantage": token_level_prefix_advantage,
        "token_adv_total_token_normalize": token_adv_total_token_normalize,
        "token_level_ndcg_error_token_penalty": token_level_ndcg_error_token_penalty,
        "fixed_hint_apply_to_eval": fixed_hint_apply_to_eval,
        "dynamic_hint_apply_to_eval": dynamic_hint_apply_to_eval,
        "eval_on_start": eval_on_start,
        "bf16": bf16,
    }
    parsed_bool_args = parse_entrypoint_bool_args(raw_bool_args, coerce_bool_arg_fn=coerce_bool_arg)
    save_only_model = parsed_bool_args.save_only_model
    do_sample = parsed_bool_args.do_sample
    prefix_reward_normalize = parsed_bool_args.prefix_reward_normalize
    probe_rule_with_zero_weight = parsed_bool_args.probe_rule_with_zero_weight
    token_level_prefix_advantage = parsed_bool_args.token_level_prefix_advantage
    token_adv_total_token_normalize = parsed_bool_args.token_adv_total_token_normalize
    token_level_ndcg_error_token_penalty = parsed_bool_args.token_level_ndcg_error_token_penalty
    fixed_hint_apply_to_eval = parsed_bool_args.fixed_hint_apply_to_eval
    dynamic_hint_apply_to_eval = parsed_bool_args.dynamic_hint_apply_to_eval
    eval_on_start = parsed_bool_args.eval_on_start
    bf16 = parsed_bool_args.bf16

    mode_config = build_trainer_mode_config(reward_mode, dynamic_hint_max_depth)
    dynamic_hint_max_depth = mode_config.dynamic_hint_max_depth
    validate_mode_configuration(
        mode_config,
        fixed_hint_depth_map_path=fixed_hint_depth_map_path,
        hint_ce_loss_coef=hint_ce_loss_coef,
    )

    train_split, eval_split, test_dataset = load_dataset_splits(
        data_dir,
        train_task_names=train_task_names,
        eval_task_names=eval_task_names,
        load_dataset_fn=load_dataset,
    )
    log_dataset_selection(train_split, eval_split, print_main_process_fn=print_main_process)
    _ = test_dataset  # noqa: F841

    tokenizer = AutoTokenizer.from_pretrained(model)
    hint_datasets = prepare_hint_datasets(
        train_split,
        eval_split,
        mode_config=mode_config,
        fixed_hint_depth_map_path=fixed_hint_depth_map_path,
        fixed_hint_depth_cap=fixed_hint_depth_cap,
        fixed_hint_unsolved_depth=fixed_hint_unsolved_depth,
        fixed_hint_task_names=fixed_hint_task_names,
        fixed_hint_apply_to_eval=fixed_hint_apply_to_eval,
        dynamic_hint_apply_to_eval=dynamic_hint_apply_to_eval,
        dynamic_hint_task_names=dynamic_hint_task_names,
        load_fixed_hint_depth_map_fn=load_fixed_hint_depth_map,
        apply_fixed_hint_depth_to_example_fn=apply_fixed_hint_depth_to_example,
        print_main_process_fn=print_main_process,
    )
    train_dataset = hint_datasets.train_dataset
    eval_dataset = hint_datasets.eval_dataset

    reward_funcs, reward_weights = build_reward_setup(
        reward_mode=reward_mode,
        num_beams=num_beams,
        prefix_reward_normalize=prefix_reward_normalize,
        probe_rule_with_zero_weight=probe_rule_with_zero_weight,
    )

    training_args = build_training_args(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_step=eval_step,
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
        num_beams=num_beams,
        bf16=bf16,
        deepspeed=deepspeed,
        report_to=report_to,
        run_name=run_name,
        hint_ce_loss_coef=hint_ce_loss_coef,
        reward_weights=reward_weights,
        get_grpo_config_fn=get_grpo_config,
    )

    logits_processor = build_logits_processor(
        index_path=index_path,
        tokenizer=tokenizer,
        prefix=prefix,
        num_beams=num_beams,
        sid_levels=sid_levels,
        use_fixed_hint_processor=fixed_hint_depth_map_path is not None or mode_config.dynamic_hint_enabled,
        build_constrained_logits_processor_fn=build_constrained_logits_processor,
        build_fixed_hint_constrained_logits_processor_fn=build_fixed_hint_constrained_logits_processor,
    )

    model = MIMIGenRec.from_pretrained(
        model,
        logits_processor=logits_processor,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_beams=num_beams,
        max_completion_length=max_completion_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )

    log_runtime_configuration(
        raw_bool_args=raw_bool_args,
        parsed_bool_args=parsed_bool_args,
        reward_mode=reward_mode,
        fixed_hint_depth_map_path=fixed_hint_depth_map_path,
        fixed_hint_depth_cap=fixed_hint_depth_cap,
        fixed_hint_unsolved_depth=fixed_hint_unsolved_depth,
        hint_ce_loss_coef=hint_ce_loss_coef,
        mode_config=mode_config,
        save_only_model=save_only_model,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        format_typed_value_fn=format_typed_value,
        print_main_process_fn=print_main_process,
    )

    trainer = build_trainer(
        fixed_hint_depth_map_path=fixed_hint_depth_map_path,
        mode_config=mode_config,
        token_level_prefix_advantage=token_level_prefix_advantage,
        model=model,
        training_args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        hint_ce_loss_coef=hint_ce_loss_coef,
        dynamic_hint_apply_to_eval=dynamic_hint_apply_to_eval,
        prefix_reward_normalize=prefix_reward_normalize,
        token_adv_total_token_normalize=token_adv_total_token_normalize,
        token_level_ndcg_error_token_penalty=token_level_ndcg_error_token_penalty,
        fixed_hint_trainer_cls=FixedHintRuleOnlyGRPOTrainer,
        dynamic_hint_trainer_cls=DynamicHintRuleOnlyGRPOTrainer,
        token_prefix_trainer_cls=TokenPrefixGRPOTrainer,
        base_trainer_cls=GRPOTrainer,
    )

    resolved_resume = resolve_resume_checkpoint_path(
        resume_from_checkpoint,
        output_dir,
        get_last_checkpoint_fn=get_last_checkpoint,
        print_main_process_fn=print_main_process,
    )

    if resolved_resume is not None:
        trainer.train(resume_from_checkpoint=resolved_resume)
    else:
        trainer.train()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenRec RL trainer CLI.")
    parser.add_argument("--config", help="Path to JSON / TOML / YAML config file.")
    parser.add_argument(
        "--set",
        dest="override_items",
        action="append",
        default=[],
        help="Arbitrary override in key=value or section.key=value form. Can be repeated.",
    )
    parser.add_argument("--print-config", action="store_true", help="Print resolved nested config and exit.")
    parser.add_argument("--print-flat-kwargs", action="store_true", help="Print resolved trl_trainer kwargs and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print config + kwargs without starting training.")
    _add_cli_override_args(parser)
    return parser


def _add_cli_override_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir")
    parser.add_argument("--index-path", "--index_path", dest="index_path")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir")
    parser.add_argument("--prefix")
    parser.add_argument("--num-beams", "--num_beams", dest="num_beams")
    parser.add_argument("--sid-levels", "--sid_levels", dest="sid_levels")
    parser.add_argument("--reward-mode", "--reward_mode", dest="reward_mode")
    parser.add_argument("--hint-mode", choices=["none", "fixed", "dynamic"])
    parser.add_argument("--fixed-hint-depth-map-path", "--fixed_hint_depth_map_path", dest="fixed_hint_depth_map_path")
    parser.add_argument("--fixed-hint-depth-cap", "--fixed_hint_depth_cap", dest="fixed_hint_depth_cap")
    parser.add_argument("--fixed-hint-unsolved-depth", "--fixed_hint_unsolved_depth", dest="fixed_hint_unsolved_depth")
    parser.add_argument("--fixed-hint-task-names", "--fixed_hint_task_names", dest="fixed_hint_task_names")
    parser.add_argument("--fixed-hint-apply-to-eval", "--fixed_hint_apply_to_eval", dest="fixed_hint_apply_to_eval")
    parser.add_argument("--dynamic-hint-max-depth", "--dynamic_hint_max_depth", dest="dynamic_hint_max_depth")
    parser.add_argument(
        "--dynamic-hint-apply-to-eval", "--dynamic_hint_apply_to_eval", dest="dynamic_hint_apply_to_eval"
    )
    parser.add_argument("--dynamic-hint-task-names", "--dynamic_hint_task_names", dest="dynamic_hint_task_names")
    parser.add_argument("--hint-ce-loss-coef", "--hint_ce_loss_coef", dest="hint_ce_loss_coef")
    parser.add_argument(
        "--token-level-prefix-advantage",
        "--token_level_prefix_advantage",
        dest="token_level_prefix_advantage",
    )
    parser.add_argument(
        "--token-adv-total-token-normalize",
        "--token_adv_total_token_normalize",
        dest="token_adv_total_token_normalize",
    )
    parser.add_argument(
        "--token-level-ndcg-error-token-penalty",
        "--token_level_ndcg_error_token_penalty",
        dest="token_level_ndcg_error_token_penalty",
    )
    parser.add_argument("--prefix-reward-normalize", "--prefix_reward_normalize", dest="prefix_reward_normalize")
    parser.add_argument(
        "--probe-rule-with-zero-weight", "--probe_rule_with_zero_weight", dest="probe_rule_with_zero_weight"
    )
    parser.add_argument("--temperature")
    parser.add_argument("--top-p", "--top_p", dest="top_p")
    parser.add_argument("--top-k", "--top_k", dest="top_k")
    parser.add_argument("--max-completion-length", "--max_completion_length", dest="max_completion_length")
    parser.add_argument("--beta")
    parser.add_argument("--repetition-penalty", "--repetition_penalty", dest="repetition_penalty")
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample")
    parser.add_argument(
        "--per-device-train-batch-size",
        "--per_device_train_batch_size",
        dest="per_device_train_batch_size",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        "--per_device_eval_batch_size",
        dest="per_device_eval_batch_size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--gradient_accumulation_steps",
        dest="gradient_accumulation_steps",
    )
    parser.add_argument("--num-train-epochs", "--num_train_epochs", dest="num_train_epochs")
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate")
    parser.add_argument("--logging-steps", "--logging_steps", dest="logging_steps")
    parser.add_argument("--eval-step", "--eval_step", dest="eval_step")
    parser.add_argument("--eval-strategy", "--eval_strategy", dest="eval_strategy")
    parser.add_argument("--eval-on-start", "--eval_on_start", dest="eval_on_start")
    parser.add_argument("--save-strategy", "--save_strategy", dest="save_strategy")
    parser.add_argument("--save-steps", "--save_steps", dest="save_steps")
    parser.add_argument("--save-total-limit", "--save_total_limit", dest="save_total_limit")
    parser.add_argument("--save-only-model", "--save_only_model", dest="save_only_model")
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio")
    parser.add_argument("--max-grad-norm", "--max_grad_norm", dest="max_grad_norm")
    parser.add_argument("--optim")
    parser.add_argument("--lr-scheduler-type", "--lr_scheduler_type", dest="lr_scheduler_type")
    parser.add_argument("--bf16")
    parser.add_argument("--deepspeed")
    parser.add_argument("--report-to", "--report_to", dest="report_to")
    parser.add_argument("--run-name", "--run_name", dest="run_name")
    parser.add_argument("--resume-from-checkpoint", "--resume_from_checkpoint", dest="resume_from_checkpoint")
    parser.add_argument("--train-task-names", "--train_task_names", dest="train_task_names")
    parser.add_argument("--eval-task-names", "--eval_task_names", dest="eval_task_names")


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "data_dir": args.data_dir,
        "index_path": args.index_path,
        "output_dir": args.output_dir,
        "prefix": args.prefix,
        "num_beams": args.num_beams,
        "sid_levels": args.sid_levels,
        "reward_mode": args.reward_mode,
        "hint_mode": args.hint_mode,
        "fixed_hint_depth_map_path": args.fixed_hint_depth_map_path,
        "fixed_hint_depth_cap": args.fixed_hint_depth_cap,
        "fixed_hint_unsolved_depth": args.fixed_hint_unsolved_depth,
        "fixed_hint_task_names": args.fixed_hint_task_names,
        "fixed_hint_apply_to_eval": args.fixed_hint_apply_to_eval,
        "dynamic_hint_max_depth": args.dynamic_hint_max_depth,
        "dynamic_hint_apply_to_eval": args.dynamic_hint_apply_to_eval,
        "dynamic_hint_task_names": args.dynamic_hint_task_names,
        "hint_ce_loss_coef": args.hint_ce_loss_coef,
        "token_level_prefix_advantage": args.token_level_prefix_advantage,
        "token_adv_total_token_normalize": args.token_adv_total_token_normalize,
        "token_level_ndcg_error_token_penalty": args.token_level_ndcg_error_token_penalty,
        "prefix_reward_normalize": args.prefix_reward_normalize,
        "probe_rule_with_zero_weight": args.probe_rule_with_zero_weight,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_completion_length": args.max_completion_length,
        "beta": args.beta,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "eval_step": args.eval_step,
        "eval_strategy": args.eval_strategy,
        "eval_on_start": args.eval_on_start,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_only_model": args.save_only_model,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "bf16": args.bf16,
        "deepspeed": args.deepspeed,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "train_task_names": args.train_task_names,
        "eval_task_names": args.eval_task_names,
    }


def main(
    argv: list[str] | None = None,
    *,
    trl_main: Callable[..., Any] | None = None,
    base_defaults: dict[str, Any] | None = None,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = RLTrainerConfig.from_sources(
        config_path=args.config,
        override_items=args.override_items,
        cli_overrides=_build_cli_overrides(args),
        trl_main=trl_main or run_training_kwargs,
        base_defaults=base_defaults,
    )
    if args.print_config or args.dry_run:
        print(json.dumps(config.to_nested_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    trl_kwargs = config.to_trl_kwargs()
    if args.print_flat_kwargs or args.dry_run:
        print(json.dumps(trl_kwargs, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    if args.print_config or args.print_flat_kwargs or args.dry_run:
        return 0
    (trl_main or run_training_kwargs)(**trl_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
