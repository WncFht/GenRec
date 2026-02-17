#!/usr/bin/env bash
set -euo pipefail

# Wrapper: run Instruments preprocess in MIMIOneRec-style split (global 8:1:1).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"

SPLIT_STRATEGY=mimionerec \
TRAIN_RATIO="${TRAIN_RATIO:-0.8}" \
VALID_RATIO="${VALID_RATIO:-0.1}" \
bash "${SCRIPT_DIR}/run_instruments_preprocess.sh" "${MODE}"

