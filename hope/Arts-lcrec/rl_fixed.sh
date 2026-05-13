#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Arts_grec_index_lcrec}"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Arts-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu/checkpoint-17268}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/Arts-grec-lcrec-fixed-from-sft17268}"
export RUN_NAME="${RUN_NAME:-arts_grec_lcrec_fixed_from_sft17268}"
export FIXED_HINT_ENABLED="${FIXED_HINT_ENABLED:-true}"
export HINT_CE_LOSS_COEF="${HINT_CE_LOSS_COEF:-0.0}"

exec bash "${REPO_ROOT}/hope/_canonical_rl_launcher.sh" "$@"
