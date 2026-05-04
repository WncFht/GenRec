#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash hope/Games/prepare_games_grec_lcrec_aligned_data.sh [check|build] [--dry-run]

Modes:
  check  Inspect resolved Games paths and stable preprocess settings
  build  Build GRec-style Games data aligned to LC-Rec-style history/max settings

Important defaults:
  - split strategy defaults to GRec
  - history_max defaults to 20 to match LC-Rec
  - train row order defaults to forward to match LC-Rec
  - seq sample defaults to 10000, consistent with current Games preprocess setup
  - RL task filters remain disabled; this is a plain LCRecAligned SFT data build
EOF
}

run_cmd() {
  printf '[CMD] '
  printf '%q ' "$@"
  echo
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
GENREC_ROOT="${GENREC_ROOT:-${REPO_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
CATEGORY="${CATEGORY:-Games}"
INDEX_PATH="${INDEX_PATH:-${DATA_ROOT}/${CATEGORY}/Games.index.json}"

MODE="check"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    check|build)
      MODE="$1"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEQ_SAMPLE="${SEQ_SAMPLE:-10000}"
SEED="${SEED:-42}"
SID_LEVELS="${SID_LEVELS:--1}"
HISTORY_MAX="${HISTORY_MAX:-20}"
TRAIN_ROW_ORDER="${TRAIN_ROW_ORDER:-forward}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-grec}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VALID_RATIO="${VALID_RATIO:-0.1}"
DATA_VARIANT="${DATA_VARIANT:-Games_grec_index}"
DATASET_SUBDIR="${DATASET_SUBDIR:-${DATA_VARIANT}}"
DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX:-Games_grec_index}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-${GENREC_ROOT}/data/dataset_info.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${GENREC_ROOT}/data/${DATA_VARIANT}}"

print_config() {
  cat <<EOF
[INFO] MODE=${MODE}
[INFO] REPO_ROOT=${REPO_ROOT}
[INFO] GENREC_ROOT=${GENREC_ROOT}
[INFO] DATA_ROOT=${DATA_ROOT}
[INFO] CATEGORY=${CATEGORY}
[INFO] INDEX_PATH=${INDEX_PATH}
[INFO] OUTPUT_DIR=${OUTPUT_DIR}
[INFO] DATASET_SUBDIR=${DATASET_SUBDIR}
[INFO] DATASET_KEY_PREFIX=${DATASET_KEY_PREFIX}
[INFO] DATASET_INFO_PATH=${DATASET_INFO_PATH}
[INFO] PYTHON_BIN=${PYTHON_BIN}
[INFO] SEQ_SAMPLE=${SEQ_SAMPLE}
[INFO] SEED=${SEED}
[INFO] SID_LEVELS=${SID_LEVELS}
[INFO] HISTORY_MAX=${HISTORY_MAX}
[INFO] TRAIN_ROW_ORDER=${TRAIN_ROW_ORDER}
[INFO] SPLIT_STRATEGY=${SPLIT_STRATEGY}
[INFO] TRAIN_RATIO=${TRAIN_RATIO}
[INFO] VALID_RATIO=${VALID_RATIO}
[INFO] DRY_RUN=${DRY_RUN}
EOF
}

check_inputs() {
  [[ -d "${GENREC_ROOT}" ]] || { echo "[ERROR] Missing GENREC_ROOT: ${GENREC_ROOT}"; exit 1; }
  [[ -f "${INDEX_PATH}" ]] || { echo "[ERROR] Missing INDEX_PATH: ${INDEX_PATH}"; exit 1; }
  [[ -f "${DATA_ROOT}/${CATEGORY}/${CATEGORY}.item.json" ]] || { echo "[ERROR] Missing item json"; exit 1; }
  [[ -f "${DATA_ROOT}/${CATEGORY}/${CATEGORY}.inter.json" ]] || { echo "[ERROR] Missing inter json"; exit 1; }
  [[ -f "${GENREC_ROOT}/scripts/run_games_preprocess.sh" ]] || { echo "[ERROR] Missing run_games_preprocess.sh"; exit 1; }
}

step_check() {
  check_inputs
  run_cmd "${PYTHON_BIN}" "${GENREC_ROOT}/scripts/inspect_preprocess_category.py" \
    --category-dir "${DATA_ROOT}/${CATEGORY}" \
    --category "${CATEGORY}" \
    --index-path "${INDEX_PATH}"
}

step_build() {
  check_inputs
  run_cmd env \
    GENREC_ROOT="${GENREC_ROOT}" \
    DATA_ROOT="${DATA_ROOT}" \
    CATEGORY="${CATEGORY}" \
    INDEX_PATH="${INDEX_PATH}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DATASET_SUBDIR="${DATASET_SUBDIR}" \
    DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX}" \
    DATASET_INFO_PATH="${DATASET_INFO_PATH}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    SEQ_SAMPLE="${SEQ_SAMPLE}" \
    SEED="${SEED}" \
    SID_LEVELS="${SID_LEVELS}" \
    HISTORY_MAX="${HISTORY_MAX}" \
    TRAIN_ROW_ORDER="${TRAIN_ROW_ORDER}" \
    SPLIT_STRATEGY="${SPLIT_STRATEGY}" \
    TRAIN_RATIO="${TRAIN_RATIO}" \
    VALID_RATIO="${VALID_RATIO}" \
    DATA_VARIANT="${DATA_VARIANT}" \
    bash "${GENREC_ROOT}/scripts/run_games_preprocess.sh" build
}

print_config

case "${MODE}" in
  check) step_check ;;
  build) step_build ;;
esac
