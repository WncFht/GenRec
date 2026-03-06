import os
from typing import Optional, Union

import fire
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer

from MIMIGenRec import MIMIGenRec, get_grpo_config
from rewards.ranking_reward import build_reward_setup
from token_prefix_grpo_trainer import TokenPrefixGRPOTrainer
from util import build_constrained_logits_processor


def main(
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
    save_strategy: str = "steps",
    save_steps: Union[int, float] = 0.1,
    save_total_limit: int = 3,
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
    bf16: bool = True,
    deepspeed: Optional[str] = None,
    report_to: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = "auto",
):
    # load dataset
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
    test_dataset = dataset["test"]  # noqa: F841

    reward_funcs, reward_weights = build_reward_setup(
        reward_mode=reward_mode,
        num_beams=num_beams,
        prefix_reward_normalize=prefix_reward_normalize,
        probe_rule_with_zero_weight=probe_rule_with_zero_weight,
    )

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
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        beta=beta,
        num_generations=num_beams,
        bf16=bf16,
        deepspeed=deepspeed,
        report_to=report_to,
    )
    if reward_weights is not None:
        grpo_config_kwargs["reward_weights"] = reward_weights
    training_args = get_grpo_config(**grpo_config_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model)
    logits_processor = build_constrained_logits_processor(
        index_path,
        tokenizer,
        prefix=prefix,
        num_beams=num_beams,
        sid_levels=sid_levels,
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

    print(
        f"[INFO] reward_mode={reward_mode}, "
        f"prefix_reward_normalize={prefix_reward_normalize}, "
        f"probe_rule_with_zero_weight={probe_rule_with_zero_weight}, "
        f"token_level_prefix_advantage={token_level_prefix_advantage}, "
        f"token_adv_total_token_normalize={token_adv_total_token_normalize}, "
        f"token_level_ndcg_error_token_penalty={token_level_ndcg_error_token_penalty}, "
        f"num_reward_funcs={len(reward_funcs)}, "
        f"reward_weights={reward_weights}"
    )

    if token_level_prefix_advantage:
        trainer = TokenPrefixGRPOTrainer(
            model=model,
            args=training_args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prefix_reward_normalize=prefix_reward_normalize,
            token_adv_total_token_normalize=token_adv_total_token_normalize,
            token_level_ndcg_error_token_penalty=token_level_ndcg_error_token_penalty,
        )
    else:
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    resolved_resume = resume_from_checkpoint
    if isinstance(resolved_resume, str):
        lowered = resolved_resume.strip().lower()
        if lowered in {"", "none", "false"}:
            resolved_resume = None
        elif lowered == "auto":
            resolved_resume = get_last_checkpoint(output_dir)
            if resolved_resume is None:
                print(f"[INFO] No checkpoint found under {output_dir}, start from scratch.")
            else:
                print(f"[INFO] Auto resume from checkpoint: {resolved_resume}")

    if resolved_resume is not None:
        if not os.path.isdir(resolved_resume):
            raise FileNotFoundError(f"Checkpoint path not found: {resolved_resume}")
        trainer.train(resume_from_checkpoint=resolved_resume)
    else:
        trainer.train()


if __name__ == "__main__":
    fire.Fire(main)


# from datasets import load_dataset
# from trl import GRPOTrainer

# dataset = load_dataset("trl-lib/tldr", split="train")

# # Dummy reward function: count the number of unique characters in the completions
# def reward_num_unique_chars(completions, **kwargs):
#     return [len(set(c)) for c in completions]

# from trl.trainer.grpo_config import GRPOConfig
# args = GRPOConfig(
#     output_dir="rl_outputs/Qwen2.5-0.5B-Instruct-grpo",
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=1,
#     learning_rate=5e-6,
#     logging_steps=10,
#     num_generations=2,
#     top_k=50,
#     top_p=1.0,
#     max_completion_length=128,
# )
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# trainer = GRPOTrainer(
#     model=model,
#     reward_funcs=reward_num_unique_chars,
#     train_dataset=dataset,
#     args=args,
# )
# trainer.train()
