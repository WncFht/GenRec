#!/usr/bin/env bash
set -eo pipefail

source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate genrec
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_LIST="${CUDA_LIST:-0 1 2 3}"

REPO_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec"
INSTANCE="${INSTANCE:-remote_eval_plan_instruments}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/evaluate_all_checkpoints}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${INSTANCE}.log}"

# Fail closed: if a model/output name is not in the manifest, do not guess.
export ALLOW_HEURISTIC_FALLBACK="${ALLOW_HEURISTIC_FALLBACK:-0}"

# Limit the dry-run planner to Instruments-related outputs by default.
export MODEL_FILTER="${MODEL_FILTER:-Instruments,ins-}"

cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"

echo "[INFO] repo_root=$REPO_ROOT"
echo "[INFO] instance=$INSTANCE"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] model_filter=$MODEL_FILTER"
echo "[INFO] launch_mode=foreground-dry-run-once"

exec > >(tee -a "$LOG_FILE") 2>&1
exec env DRY_RUN=1 bash "$REPO_ROOT/scripts/evaluate_all_checkpoints.sh" once
