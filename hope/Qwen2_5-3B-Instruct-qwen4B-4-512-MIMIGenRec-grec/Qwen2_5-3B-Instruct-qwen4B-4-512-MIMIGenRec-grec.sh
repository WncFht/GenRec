#!/usr/bin/env bash
set -eo pipefail

source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate genrec
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++

REPO_ROOT="${REPO_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec}"
YAML_PATH="${YAML_PATH:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/examples/train_full/Instruments/instruments_rec_full_sft_3b_dsz0_qwen4b_4_512_grec.yaml}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/dataset_info.json}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"
  exit 1
fi
cd "${REPO_ROOT}"

export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# offline mode (use local checkpoints/datasets)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-SFT}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

if [[ ! -f "${YAML_PATH}" ]]; then
  echo "[ERROR] YAML not found: ${YAML_PATH}"
  exit 1
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "[ERROR] llamafactory-cli not found in PATH"
  exit 1
fi

if [[ ! -f "${DATASET_INFO_PATH}" ]]; then
  echo "[ERROR] dataset_info not found: ${DATASET_INFO_PATH}"
  exit 1
fi

sanitize_log_name() {
  local v="$1"
  v="${v//\//_}"
  v="${v// /_}"
  echo "$v"
}

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log}"
RUN_NAME="${RUN_NAME:-instruments_rec_full_sft_3b_dsz0_qwen4b_4_512_grec}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/$(sanitize_log_name "${RUN_NAME}")_${TS}.log"
mkdir -p "${LOG_DIR}"

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
