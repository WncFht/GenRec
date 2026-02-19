#!/usr/bin/env bash
set -eo pipefail

source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate genrec
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++

REPO_ROOT="${REPO_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/evaluate_sft_3b.sh}"

DATA_VARIANT_DEFAULT="Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb64-64-64-64_dsInstruments_ridFeb-10-2026-06-04-11"
SFT_ROOT="${SFT_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-64-dsz0}"
TEST_DATA_PATH="${TEST_DATA_PATH:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/${DATA_VARIANT_DEFAULT}/sft/test.json}"
INDEX_PATH="${INDEX_PATH:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/${DATA_VARIANT_DEFAULT}/id2sid.json}"

export CATEGORY="${CATEGORY:-Instruments_grec}"
export CUDA_LIST="${CUDA_LIST:-0 1 2 3}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export NUM_BEAMS="${NUM_BEAMS:-50}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export DO_SAMPLE="${DO_SAMPLE:-False}"
export LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"
export SID_LEVELS="${SID_LEVELS:--1}"
# optional manual checkpoint list:
#   CKPT_LIST="checkpoint-495 checkpoint-630"
#   CKPT_LIST="/abs/path/to/checkpoint-495 /abs/path/to/checkpoint-630"
export CKPT_LIST="${CKPT_LIST:-}"
export TEST_DATA_PATH
export INDEX_PATH

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"
  exit 1
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[ERROR] evaluate script not found: ${EVAL_SCRIPT}"
  exit 1
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] EVAL_SCRIPT=${EVAL_SCRIPT}"
echo "[INFO] SFT_ROOT=${SFT_ROOT}"
echo "[INFO] TEST_DATA_PATH=${TEST_DATA_PATH}"
echo "[INFO] INDEX_PATH=${INDEX_PATH}"
echo "[INFO] CUDA_LIST=${CUDA_LIST}"

cd "${REPO_ROOT}"
exec bash "${EVAL_SCRIPT}" "${SFT_ROOT}"
