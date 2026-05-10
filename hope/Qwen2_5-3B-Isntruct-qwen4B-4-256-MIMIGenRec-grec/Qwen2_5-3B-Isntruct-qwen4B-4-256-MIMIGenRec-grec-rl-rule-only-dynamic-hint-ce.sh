#!/usr/bin/env bash
set -eo pipefail
# Thin wrapper: delegates RL execution to trl_trainer.py via the shared launcher.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_rl_variant_common.sh"

DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
CONFIG_PATH_DEFAULT="${REPO_ROOT}/configs/rl/instruments/dynamic_hint_rule_ce.json"
DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PER_DEVICE_TRAIN_BSZ="${PER_DEVICE_TRAIN_BSZ:-64}"
PER_DEVICE_EVAL_BSZ="${PER_DEVICE_EVAL_BSZ:-64}"
GRAD_ACC="${GRAD_ACC:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
EVAL_ON_START="${EVAL_ON_START:-false}"
HINT_CE_LOSS_COEF="${HINT_CE_LOSS_COEF:-0.005}"

run_rl_variant_wrapper \
  --config "${CONFIG_PATH:-$CONFIG_PATH_DEFAULT}" \
  --output-dir "${REPO_ROOT}/rl_outputs/Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-hintce005-qwen2.5-3b-qwen4B-4-256-from-sft495" \
  --run_name "instruments_grec_rl_rule_only_dynamic_hint_cascade_reward_gather_fix_hint_ce005_qwen2_5_3b_qwen4b_4_256_from_ckpt495" \
  --port 29521 \
  --reward_mode rule_only \
  --dynamic_hint_max_depth 3 \
  --hint_ce_loss_coef "$HINT_CE_LOSS_COEF" \
  --eval_on_start "$EVAL_ON_START" \
  "$@"
