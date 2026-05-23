#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Instruments_grec_index}"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu/checkpoint-2751}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/Instruments-grec-genrec-fixed-ce-dual-from-sft}"
export RUN_NAME="${RUN_NAME:-instruments_grec_genrec_fixed_ce_dual_from_sft}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export MAIN_PORT="${MAIN_PORT:-29516}"
export GRAD_ACC="${GRAD_ACC:-2}"
export DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FIXED_HINT_ENABLED="${FIXED_HINT_ENABLED:-true}"
export HINT_CE_LOSS_COEF="${HINT_CE_LOSS_COEF:-0.005}"
export TRAIN_TASK_NAMES="${TRAIN_TASK_NAMES:-task1_sid_sft,task5_title_desc2sid}"
export EVAL_TASK_NAMES="${EVAL_TASK_NAMES:-task1_sid_sft}"
export ANALYSIS_TASK_NAMES="${ANALYSIS_TASK_NAMES:-${TRAIN_TASK_NAMES}}"

exec bash "${REPO_ROOT}/hope/_canonical_rl_launcher.sh" "$@"
