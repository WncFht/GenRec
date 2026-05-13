#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export YAML_PATH="${YAML_PATH:-${REPO_ROOT}/examples/train_full/Arts/arts_rec_full_sft_3b_dsz3_qwen4b_4_256_grec_lcrec_aligned_4gpu.yaml}"
export RUN_NAME="${RUN_NAME:-Arts-grec-lcrec-aligned-qwen2.5-3b-sft-qwen4B-4-256-dsz3-4gpu}"
export DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

exec bash "${REPO_ROOT}/hope/_canonical_sft_launcher.sh"
