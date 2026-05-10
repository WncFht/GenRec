#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_rl_variant_common.sh"

DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
EVAL_ON_START="${EVAL_ON_START:-true}"

run_rl_variant_wrapper \
  --config "${CONFIG_PATH:-${REPO_ROOT}/configs/rl/instruments/prefix_token_totalnorm.json}" \
  --output_dir "${REPO_ROOT}/rl_outputs/Instruments-grec-grpo-prefix-tokenadv-totalnorm-ndcg-rule0-qwen2.5-3b-qwen4B-4-256-from-sft495" \
  --run_name "instruments_grec_rl_prefix_tokenadv_totalnorm_ndcg_rule0_qwen2_5_3b_qwen4b_4_256_from_ckpt495" \
  --port 29514 \
  --eval_on_start "$EVAL_ON_START" \
  "$@"
