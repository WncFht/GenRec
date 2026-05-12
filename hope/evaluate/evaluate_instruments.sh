#!/usr/bin/env bash
set -eo pipefail

source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate genrec
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_LIST="${CUDA_LIST:-0 1 2 3}"

REPO_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec"
INSTANCE="${INSTANCE:-remote_eval_instruments}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/evaluate_all_checkpoints}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${INSTANCE}.log}"

# Fail closed: if a model/output name is not in the manifest, do not guess.
export ALLOW_HEURISTIC_FALLBACK="${ALLOW_HEURISTIC_FALLBACK:-0}"

# Keep the watcher alive on reserved GPUs, but use a slightly lower idle
# reservation ratio now that release is handled explicitly before execution.
export IDLE_HOLD_ENABLED="${IDLE_HOLD_ENABLED:-1}"
export IDLE_HOLD_MEMORY_RATIO="${IDLE_HOLD_MEMORY_RATIO:-0.90}"
export IDLE_HOLD_RELEASE_GRACE_SECONDS="${IDLE_HOLD_RELEASE_GRACE_SECONDS:-10}"
export POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"
export STABLE_AGE_SECONDS="${STABLE_AGE_SECONDS:-180}"
export STABLE_CONFIRMATION_POLLS="${STABLE_CONFIRMATION_POLLS:-2}"

# Limit the one-shot eval worker to Instruments-related outputs by default.
export MODEL_FILTER="${MODEL_FILTER:-Instruments,ins-}"

cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"

echo "[INFO] repo_root=$REPO_ROOT"
echo "[INFO] instance=$INSTANCE"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] model_filter=$MODEL_FILTER"
echo "[INFO] launch_mode=foreground-once"

exec > >(tee -a "$LOG_FILE") 2>&1
exec bash "$REPO_ROOT/scripts/evaluate_all_checkpoints.sh" once
