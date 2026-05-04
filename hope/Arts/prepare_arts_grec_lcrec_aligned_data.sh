#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash hope/Arts/prepare_arts_grec_lcrec_aligned_data.sh [check|build] [--dry-run]

Modes:
  check  Inspect resolved Arts raw inputs and target output paths
  build  Build GRec-style Arts data aligned to LC-Rec-style history/max settings

Important defaults:
  - split strategy defaults to GRec
  - history_max defaults to 20 to match LC-Rec
  - train row order defaults to forward to match LC-Rec
  - seq sample defaults to 10000
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
CATEGORY="${CATEGORY:-Arts}"
INDEX_PATH="${INDEX_PATH:-${DATA_ROOT}/${CATEGORY}/Arts.index.json}"

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
DATA_VARIANT="${DATA_VARIANT:-Arts_grec_index}"
DATASET_SUBDIR="${DATASET_SUBDIR:-${DATA_VARIANT}}"
DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX:-Arts_grec_index}"
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
  [[ -f "${GENREC_ROOT}/scripts/prepare_category_from_inter_json.py" ]] || { echo "[ERROR] Missing prepare_category_from_inter_json.py"; exit 1; }
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
  run_cmd "${PYTHON_BIN}" "${GENREC_ROOT}/scripts/prepare_category_from_inter_json.py" \
    --genrec-root "${GENREC_ROOT}" \
    --category-dir "${DATA_ROOT}/${CATEGORY}" \
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
    --history-max "${HISTORY_MAX}" \
    --train-row-order "${TRAIN_ROW_ORDER}" \
    --python-bin "${PYTHON_BIN}"
}

print_config

case "${MODE}" in
  check) step_check ;;
  build) step_build ;;
esac
