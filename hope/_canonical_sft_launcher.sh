#!/usr/bin/env bash
set -eo pipefail

require_dir() {
  local path="$1"
  local desc="$2"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing ${desc}: $path"
    exit 1
  fi
}

require_file() {
  local path="$1"
  local desc="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing ${desc}: $path"
    exit 1
  fi
}

sanitize_log_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  echo "$value"
}

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-genrec}"
REPO_ROOT="${REPO_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec}"
YAML_PATH="${YAML_PATH:-}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-${REPO_ROOT}/data/dataset_info.json}"
RUN_NAME="${RUN_NAME:-}"
DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

require_dir "$REPO_ROOT" "REPO_ROOT"
if [[ -z "$YAML_PATH" ]]; then
  echo "[ERROR] YAML_PATH is required"
  exit 1
fi
require_file "$YAML_PATH" "YAML config"
require_file "$DATASET_INFO_PATH" "dataset_info.json"
require_file "$CONDA_ACTIVATE" "conda activate script"

# shellcheck disable=SC1090
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
  source "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
fi
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
fi

cd "$REPO_ROOT"

export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-SFT}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "[ERROR] llamafactory-cli not found in PATH"
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(basename "${YAML_PATH%.yaml}")"
fi

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(sanitize_log_name "${RUN_NAME}")_${TS}.log}"
mkdir -p "$LOG_DIR"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] YAML_PATH=${YAML_PATH}"
echo "[INFO] DATASET_INFO_PATH=${DATASET_INFO_PATH}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] Launch mode=background+wait"

set +e
(
  set -o pipefail
  set -x
  llamafactory-cli train "${YAML_PATH}" 2>&1 | tee -a "${LOG_FILE}"
) &
TRAIN_PID=$!
set -e

echo "[INFO] Training started. pid=${TRAIN_PID}"
echo "[INFO] Streaming logs to terminal and ${LOG_FILE}"

set +e
wait "${TRAIN_PID}"
EXIT_CODE=$?
set -e
echo "[INFO] Training finished with exit code=${EXIT_CODE}"
exit "${EXIT_CODE}"
