#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Games_grec_index_lcrec}"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Games-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/Games-grec-lcrec-ndcg-from-sft}"
export RUN_NAME="${RUN_NAME:-games_grec_lcrec_ndcg_from_sft}"
export MAIN_PORT="${MAIN_PORT:-29517}"
export REWARD_MODE="${REWARD_MODE:-ranking}"

exec bash "${REPO_ROOT}/hope/_canonical_rl_launcher.sh" "$@"
