#!/usr/bin/env bash
set -euo pipefail

# Step-by-step runner for Instruments preprocessing on remote training machine.
# Modes:
#   check   : verify paths and inspect files only (default)
#   prepare : create .train/.valid/.test.inter only
#   build   : run full preprocess_data_sft_rl.py (expects staging to be creatable)
#   all     : check + build

# Default paths for remote machine. All can be overridden by env vars.
GENREC_ROOT="${GENREC_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec}"
DATA_ROOT="${DATA_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/data}"
CATEGORY="${CATEGORY:-Instruments}"

INDEX_BASENAME_DEFAULT="Instruments.index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47.json"
INDEX_PATH="${INDEX_PATH:-${DATA_ROOT}/${CATEGORY}/${INDEX_BASENAME_DEFAULT}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEQ_SAMPLE="${SEQ_SAMPLE:-10000}"
SEED="${SEED:-42}"
SID_LEVELS="${SID_LEVELS:--1}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-mimionerec}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VALID_RATIO="${VALID_RATIO:-0.1}"
MODE="${1:-check}"

CATEGORY_DIR="${DATA_ROOT}/${CATEGORY}"
INDEX_FILE_NAME="$(basename "${INDEX_PATH}")"
INDEX_STEM="${INDEX_FILE_NAME%.json}"
if [[ "${INDEX_STEM}" == "${CATEGORY}."* ]]; then
  INDEX_STEM="${INDEX_STEM#${CATEGORY}.}"
fi

DATA_VARIANT="${DATA_VARIANT:-${CATEGORY}_${SPLIT_STRATEGY}_${INDEX_STEM}}"
OUTPUT_DIR="${OUTPUT_DIR:-${GENREC_ROOT}/data/${DATA_VARIANT}}"
DATASET_SUBDIR="${DATASET_SUBDIR:-${DATA_VARIANT}}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-${GENREC_ROOT}/data/dataset_info.json}"
DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX:-$(echo "${DATA_VARIANT}" | sed -E 's/[^A-Za-z0-9_]+/_/g; s/_+/_/g; s/^_+//; s/_+$//')}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_instruments_preprocess.sh [check|prepare|build|all]

Examples:
  bash scripts/run_instruments_preprocess.sh check
  bash scripts/run_instruments_preprocess.sh prepare
  bash scripts/run_instruments_preprocess.sh build
  bash scripts/run_instruments_preprocess.sh all

Override env vars if needed:
  GENREC_ROOT DATA_ROOT CATEGORY INDEX_PATH OUTPUT_DIR PYTHON_BIN SEQ_SAMPLE SEED
  SID_LEVELS SPLIT_STRATEGY TRAIN_RATIO VALID_RATIO
  DATA_VARIANT DATASET_SUBDIR DATASET_KEY_PREFIX DATASET_INFO_PATH
EOF
}

print_config() {
  echo "[INFO] MODE=${MODE}"
  echo "[INFO] GENREC_ROOT=${GENREC_ROOT}"
  echo "[INFO] DATA_ROOT=${DATA_ROOT}"
  echo "[INFO] CATEGORY=${CATEGORY}"
  echo "[INFO] CATEGORY_DIR=${CATEGORY_DIR}"
  echo "[INFO] INDEX_PATH=${INDEX_PATH}"
  echo "[INFO] INDEX_STEM=${INDEX_STEM}"
  echo "[INFO] DATA_VARIANT=${DATA_VARIANT}"
  echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
  echo "[INFO] DATASET_SUBDIR=${DATASET_SUBDIR}"
  echo "[INFO] DATASET_KEY_PREFIX=${DATASET_KEY_PREFIX}"
  echo "[INFO] DATASET_INFO_PATH=${DATASET_INFO_PATH}"
  echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
  echo "[INFO] SEQ_SAMPLE=${SEQ_SAMPLE}"
  echo "[INFO] SEED=${SEED}"
  echo "[INFO] SID_LEVELS=${SID_LEVELS}"
  echo "[INFO] SPLIT_STRATEGY=${SPLIT_STRATEGY}"
  echo "[INFO] TRAIN_RATIO=${TRAIN_RATIO}"
  echo "[INFO] VALID_RATIO=${VALID_RATIO}"
}

require_dir() {
  local path="$1"
  local desc="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] ${desc} not found: ${path}"
    exit 1
  fi
}

require_file() {
  local path="$1"
  local desc="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] ${desc} not found: ${path}"
    exit 1
  fi
}

check_base() {
  require_dir "${GENREC_ROOT}" "GENREC_ROOT"
  require_dir "${CATEGORY_DIR}" "CATEGORY_DIR"
  require_file "${INDEX_PATH}" "INDEX_PATH"
  require_file "${GENREC_ROOT}/scripts/inspect_preprocess_category.py" "inspect script"
  require_file "${GENREC_ROOT}/scripts/prepare_category_from_inter_json.py" "prepare script"
  require_file "${GENREC_ROOT}/preprocess_data_sft_rl.py" "preprocess script"
}

step_check() {
  echo "[STEP] Check current paths and files"
  echo "[INFO] pwd: $(pwd)"
  echo "[INFO] whoami: $(whoami)"
  ls -ld "${GENREC_ROOT}" "${DATA_ROOT}" "${CATEGORY_DIR}"
  ls -lh "${INDEX_PATH}"

  echo "[STEP] List key raw files under ${CATEGORY_DIR}"
  for f in \
    "${CATEGORY_DIR}/${CATEGORY}.item.json" \
    "${CATEGORY_DIR}/${CATEGORY}.inter.json" \
    "${CATEGORY_DIR}/${CATEGORY}.item_enriched.json" \
    "${CATEGORY_DIR}/${CATEGORY}.item2id"; do
    if [[ -f "${f}" ]]; then
      ls -lh "${f}"
    else
      echo "[WARN] Missing: ${f}"
    fi
  done

  echo "[STEP] Show top-level files (first 40)"
  ls -1 "${CATEGORY_DIR}" | head -n 40

  echo "[STEP] Inspect content summary"
  "${PYTHON_BIN}" "${GENREC_ROOT}/scripts/inspect_preprocess_category.py" \
    --category-dir "${CATEGORY_DIR}" \
    --category "${CATEGORY}" \
    --index-path "${INDEX_PATH}"
}

step_prepare() {
  echo "[STEP] Prepare .train/.valid/.test.inter only"
  "${PYTHON_BIN}" "${GENREC_ROOT}/scripts/prepare_category_from_inter_json.py" \
    --genrec-root "${GENREC_ROOT}" \
    --category-dir "${CATEGORY_DIR}" \
    --category "${CATEGORY}" \
    --index-path "${INDEX_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --dataset-subdir "${DATASET_SUBDIR}" \
    --dataset-key-prefix "${DATASET_KEY_PREFIX}" \
    --dataset-info-path "${DATASET_INFO_PATH}" \
    --split-strategy "${SPLIT_STRATEGY}" \
    --train-ratio "${TRAIN_RATIO}" \
    --valid-ratio "${VALID_RATIO}" \
    --seq-sample "${SEQ_SAMPLE}" \
    --seed "${SEED}" \
    --sid-levels "${SID_LEVELS}" \
    --python-bin "${PYTHON_BIN}" \
    --skip-dataset-info-update \
    --prepare-only

  echo "[DONE] Prepared staging inter files under ${GENREC_ROOT}/data/_preprocess_input/${CATEGORY}"
}

step_build() {
  echo "[STEP] Build final SFT/RL dataset"
  "${PYTHON_BIN}" "${GENREC_ROOT}/scripts/prepare_category_from_inter_json.py" \
    --genrec-root "${GENREC_ROOT}" \
    --category-dir "${CATEGORY_DIR}" \
    --category "${CATEGORY}" \
    --index-path "${INDEX_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --dataset-subdir "${DATASET_SUBDIR}" \
    --dataset-key-prefix "${DATASET_KEY_PREFIX}" \
    --dataset-info-path "${DATASET_INFO_PATH}" \
    --split-strategy "${SPLIT_STRATEGY}" \
    --train-ratio "${TRAIN_RATIO}" \
    --valid-ratio "${VALID_RATIO}" \
    --seq-sample "${SEQ_SAMPLE}" \
    --seed "${SEED}" \
    --sid-levels "${SID_LEVELS}" \
    --python-bin "${PYTHON_BIN}"

  echo "[DONE] Output directory: ${OUTPUT_DIR}"
  echo "[DONE] Dataset keys: ${DATASET_KEY_PREFIX}_train, ${DATASET_KEY_PREFIX}_valid"
  echo "[DONE] SFT train: ${OUTPUT_DIR}/sft/train.json"
  echo "[DONE] RL train: ${OUTPUT_DIR}/rl/train.json"
  echo "[DONE] New tokens: ${OUTPUT_DIR}/new_tokens.json"
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" || "${MODE}" == "help" ]]; then
  usage
  exit 0
fi

cd "${GENREC_ROOT}"
print_config
check_base

case "${MODE}" in
  check)
    step_check
    ;;
  prepare)
    step_prepare
    ;;
  build)
    step_build
    ;;
  all)
    step_check
    step_build
    ;;
  *)
    echo "[ERROR] Unknown mode: ${MODE}"
    usage
    exit 1
    ;;
esac
