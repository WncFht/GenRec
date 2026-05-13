#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export GENREC_ROOT="${GENREC_ROOT:-${REPO_ROOT}}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export CATEGORY="${CATEGORY:-Arts}"
export INDEX_PATH="${INDEX_PATH:-${DATA_ROOT}/${CATEGORY}/Arts.index.json}"
export DATA_VARIANT="${DATA_VARIANT:-Arts_grec_index}"
export DATASET_SUBDIR="${DATASET_SUBDIR:-Arts_grec_index}"
export DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX:-Arts_grec_index}"
export OUTPUT_DIR="${OUTPUT_DIR:-${GENREC_ROOT}/data/Arts_grec_index}"
export HISTORY_MAX="${HISTORY_MAX:-20}"
export TRAIN_ROW_ORDER="${TRAIN_ROW_ORDER:-forward}"
export SPLIT_STRATEGY="${SPLIT_STRATEGY:-grec}"

exec bash "${REPO_ROOT}/hope/Arts/prepare_arts_grec_lcrec_aligned_data.sh" "$@"
