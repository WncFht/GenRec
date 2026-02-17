#!/usr/bin/env bash
set -euo pipefail

# Wrapper: run Instruments preprocess in GRec-style split (per-user leave-2-out).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"

SPLIT_STRATEGY=grec \
bash "${SCRIPT_DIR}/run_instruments_preprocess.sh" "${MODE}"

