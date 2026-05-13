#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export GENREC_ROOT="${GENREC_ROOT:-${REPO_ROOT}}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export CATEGORY="${CATEGORY:-Instruments}"
export INDEX_PATH="${INDEX_PATH:-${DATA_ROOT}/${CATEGORY}/${CATEGORY}.index.json}"
export DATA_VARIANT="${DATA_VARIANT:-Instruments_grec_index}"
export DATASET_SUBDIR="${DATASET_SUBDIR:-Instruments_grec_index}"
export DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX:-Instruments_grec_index}"
export OUTPUT_DIR="${OUTPUT_DIR:-${GENREC_ROOT}/data/Instruments_grec_index}"
export SEQ_SAMPLE="${SEQ_SAMPLE:-10000}"
export TASK3_SAMPLE="${TASK3_SAMPLE:-20000}"
export HISTORY_MAX="${HISTORY_MAX:-50}"
export TRAIN_ROW_ORDER="${TRAIN_ROW_ORDER:-reverse}"
export SPLIT_STRATEGY="${SPLIT_STRATEGY:-grec}"

exec bash "${REPO_ROOT}/hope/Instruments/prepare_instruments_grec_sft_fixed_hint_data.sh" "$@"
