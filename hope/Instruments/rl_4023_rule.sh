#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_SCRIPT="${SCRIPT_DIR}/../Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh"

MODEL_PATH_DEFAULT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Instruments-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu/checkpoint-4023"
DATA_VARIANT_DEFAULT="Instruments_grec_index"
DATA_DIR_DEFAULT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/${DATA_VARIANT_DEFAULT}/rl"
INDEX_PATH_DEFAULT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/${DATA_VARIANT_DEFAULT}/id2sid.json"
OUTPUT_DIR_DEFAULT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/rl_outputs/Instruments-grec-lc4023-rule"
RUN_NAME_DEFAULT="instruments_grec_lc4023_rule"

exec bash "${ROOT_SCRIPT}" \
  --model-path "${MODEL_PATH:-${MODEL_PATH_DEFAULT}}" \
  --data-dir "${DATA_DIR:-${DATA_DIR_DEFAULT}}" \
  --index-path "${INDEX_PATH:-${INDEX_PATH_DEFAULT}}" \
  --output-dir "${OUTPUT_DIR:-${OUTPUT_DIR_DEFAULT}}" \
  --run-name "${RUN_NAME:-${RUN_NAME_DEFAULT}}" \
  --resume "${RESUME_FROM_CHECKPOINT:-auto}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-10}" \
  "$@"
