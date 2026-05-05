#!/usr/bin/env bash
set -eo pipefail

source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate genrec
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_LIST="${CUDA_LIST:-0 1 2 3}"

REPO_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec"
INSTANCE="${INSTANCE:-remote_eval}"

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

cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/evaluate_all_checkpoints.sh" start --instance "$INSTANCE"
bash "$REPO_ROOT/scripts/evaluate_all_checkpoints.sh" status --instance "$INSTANCE"
