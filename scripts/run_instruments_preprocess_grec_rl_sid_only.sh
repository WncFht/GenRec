#!/usr/bin/env bash
set -eo pipefail

# Wrapper: build the Instruments GRec split, but keep only task1 SID samples in RL output.
# SFT generation remains unchanged.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-check}"

SPLIT_STRATEGY=grec \
DATA_VARIANT_TAG=rlsidonly \
RL_ONLY_TASK1=true \
bash "${SCRIPT_DIR}/run_instruments_preprocess.sh" "${MODE}"
