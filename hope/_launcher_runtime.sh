#!/usr/bin/env bash

activate_genrec_env() {
  local conda_activate="$1"
  local conda_env_name="$2"

  if [[ ! -f "$conda_activate" ]]; then
    echo "[ERROR] Conda activate script not found: $conda_activate"
    exit 1
  fi

  # shellcheck disable=SC1090
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$conda_env_name" ]]; then
    source "$conda_activate" "$conda_env_name"
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
  fi
}

setup_genrec_runtime_env() {
  local repo_root="$1"

  export DISABLE_VERSION_CHECK=1
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  unset HF_ENDPOINT || true
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
  export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-warning}"
  export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
}

require_accelerate() {
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] accelerate not found in PATH"
    exit 1
  fi
}
