#!/usr/bin/env bash
set -eo pipefail

# Wrapper: run Games preprocess in GRec-style split (per-user leave-2-out).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"
DEFAULT_GENREC_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec"
STABLE_VARIANT="${DATA_VARIANT:-Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames}"
STABLE_OUTPUT_DIR="${OUTPUT_DIR:-${GENREC_ROOT:-$DEFAULT_GENREC_ROOT}/data/${STABLE_VARIANT}}"

SPLIT_STRATEGY=grec \
DATA_VARIANT="${STABLE_VARIANT}" \
OUTPUT_DIR="${STABLE_OUTPUT_DIR}" \
DATASET_SUBDIR="${DATASET_SUBDIR:-${STABLE_VARIANT}}" \
DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX:-Games_grec_index_emb_qwen3_embedding_4B_rq4_cb256_256_256_256_dsGames}" \
bash "${SCRIPT_DIR}/run_games_preprocess.sh" "${MODE}"
