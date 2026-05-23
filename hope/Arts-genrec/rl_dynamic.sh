#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Arts_grec_index}"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Arts-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/Arts-grec-genrec-dynamic-hint-from-sft}"
export RUN_NAME="${RUN_NAME:-arts_grec_genrec_dynamic_hint_from_sft}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export MAIN_PORT="${MAIN_PORT:-29516}"
export GRAD_ACC="${GRAD_ACC:-2}"
export DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DYNAMIC_HINT_MAX_DEPTH="${DYNAMIC_HINT_MAX_DEPTH:-3}"
export DYNAMIC_HINT_APPLY_TO_EVAL="${DYNAMIC_HINT_APPLY_TO_EVAL:-false}"
export HINT_CE_LOSS_COEF="${HINT_CE_LOSS_COEF:-0.0}"

exec bash "${REPO_ROOT}/hope/_canonical_rl_launcher.sh" "$@"
