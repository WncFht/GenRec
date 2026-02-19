#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash Qwen2_5-3B-Instruct-qwen4B-4-512-MIMIGenRec-grec-rl.sh [options]

Run modes:
  --nohup                 Start RL in background via nohup and follow log (default)
  --detach                Start RL in background via nohup and do not follow log
  --tail                  Follow latest log for current run name
  --run                   Internal mode; run training command directly

Common overrides:
  --model-path <path>
  --data-dir <path>
  --index-path <path>
  --output-dir <path>
  --resume <auto|none|path>
  --run-name <name>
  --num-processes <n>
  --port <n>
  --ds-config <path>
  --num-beams <n>
  --sid-levels <n>
  --train-bsz <n>
  --eval-bsz <n>
  --grad-acc <n>
  --epochs <n>
  --lr <float>
  --eval-step <n>
  --max-completion-length <n>
  --beta <float>
  --temperature <float>
  --save-total-limit <n>
  --report-to <name>
  --wandb-mode <offline|online|disabled>
  --log-dir <path>
  --log-file <path>
  --dry-run
  -h, --help
EOF
}

sanitize_log_name() {
  local v="$1"
  v="${v//\//_}"
  v="${v// /_}"
  echo "$v"
}

require_exists() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "$path" ]]; then
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

latest_log_for_prefix() {
  local log_dir="$1"
  local prefix="$2"
  local latest=""
  latest="$(ls -1t "${log_dir}/$(sanitize_log_name "${prefix}")"_*.log 2>/dev/null | head -n 1 || true)"
  echo "$latest"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

MODE="nohup" # nohup | detach | tail | run
DRY_RUN=0
FROM_NOHUP=0
LOG_FILE_OVERRIDE=""
FORWARD_ARGS=()

# cluster/env defaults
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-genrec}"
REPO_ROOT="${REPO_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec}"
DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb512-512-512-512_dsInstruments_ridFeb-10-2026-03-29-55}"

# continue from provided SFT checkpoint-495 by default
MODEL_PATH="${MODEL_PATH:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-512-dsz0/checkpoint-495}"
DATA_DIR="${DATA_DIR:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/${DATA_VARIANT_DEFAULT}/rl}"
INDEX_PATH="${INDEX_PATH:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/data/${DATA_VARIANT_DEFAULT}/id2sid.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/rl_outputs/Instruments-grec-grpo-qwen2.5-3b-qwen4B-4-512-from-sft495}"
DS_CONFIG="${DS_CONFIG:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/config/zero2.yaml}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CATEGORY="${CATEGORY:-Instruments_grec}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PORT="${MAIN_PORT:-29513}"
NUM_BEAMS="${NUM_BEAMS:-16}"
SID_LEVELS="${SID_LEVELS:--1}"
PER_DEVICE_TRAIN_BSZ="${PER_DEVICE_TRAIN_BSZ:-64}"
PER_DEVICE_EVAL_BSZ="${PER_DEVICE_EVAL_BSZ:-64}"
GRAD_ACC="${GRAD_ACC:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EVAL_STEP="${EVAL_STEP:-20}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
BETA="${BETA:-1e-3}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
REPORT_TO="${REPORT_TO:-wandb}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"

export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-instruments_grec_rl_qwen2_5_3b_qwen4b_4_512_from_ckpt495}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nohup)
      MODE="nohup"
      shift
      ;;
    --detach)
      MODE="detach"
      shift
      ;;
    --tail)
      MODE="tail"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --from-nohup)
      FROM_NOHUP=1
      shift
      ;;
    --model-path)
      MODEL_PATH="$2"
      FORWARD_ARGS+=("--model-path" "$2")
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      FORWARD_ARGS+=("--data-dir" "$2")
      shift 2
      ;;
    --index-path)
      INDEX_PATH="$2"
      FORWARD_ARGS+=("--index-path" "$2")
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      FORWARD_ARGS+=("--output-dir" "$2")
      shift 2
      ;;
    --resume|--resume-from-checkpoint)
      RESUME_FROM_CHECKPOINT="$2"
      FORWARD_ARGS+=("--resume" "$2")
      shift 2
      ;;
    --run-name)
      export WANDB_RUN_NAME="$2"
      FORWARD_ARGS+=("--run-name" "$2")
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES="$2"
      FORWARD_ARGS+=("--num-processes" "$2")
      shift 2
      ;;
    --port)
      MAIN_PORT="$2"
      FORWARD_ARGS+=("--port" "$2")
      shift 2
      ;;
    --ds-config)
      DS_CONFIG="$2"
      FORWARD_ARGS+=("--ds-config" "$2")
      shift 2
      ;;
    --num-beams)
      NUM_BEAMS="$2"
      FORWARD_ARGS+=("--num-beams" "$2")
      shift 2
      ;;
    --sid-levels)
      SID_LEVELS="$2"
      FORWARD_ARGS+=("--sid-levels" "$2")
      shift 2
      ;;
    --train-bsz)
      PER_DEVICE_TRAIN_BSZ="$2"
      FORWARD_ARGS+=("--train-bsz" "$2")
      shift 2
      ;;
    --eval-bsz)
      PER_DEVICE_EVAL_BSZ="$2"
      FORWARD_ARGS+=("--eval-bsz" "$2")
      shift 2
      ;;
    --grad-acc)
      GRAD_ACC="$2"
      FORWARD_ARGS+=("--grad-acc" "$2")
      shift 2
      ;;
    --epochs)
      NUM_EPOCHS="$2"
      FORWARD_ARGS+=("--epochs" "$2")
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      FORWARD_ARGS+=("--lr" "$2")
      shift 2
      ;;
    --eval-step)
      EVAL_STEP="$2"
      FORWARD_ARGS+=("--eval-step" "$2")
      shift 2
      ;;
    --max-completion-length)
      MAX_COMPLETION_LENGTH="$2"
      FORWARD_ARGS+=("--max-completion-length" "$2")
      shift 2
      ;;
    --beta)
      BETA="$2"
      FORWARD_ARGS+=("--beta" "$2")
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      FORWARD_ARGS+=("--temperature" "$2")
      shift 2
      ;;
    --save-total-limit)
      SAVE_TOTAL_LIMIT="$2"
      FORWARD_ARGS+=("--save-total-limit" "$2")
      shift 2
      ;;
    --report-to)
      REPORT_TO="$2"
      FORWARD_ARGS+=("--report-to" "$2")
      shift 2
      ;;
    --wandb-mode)
      export WANDB_MODE="$2"
      FORWARD_ARGS+=("--wandb-mode" "$2")
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE_OVERRIDE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      FORWARD_ARGS+=("--dry-run")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$LOG_FILE_OVERRIDE" ]]; then
  LOG_FILE="$LOG_FILE_OVERRIDE"
else
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/$(sanitize_log_name "${WANDB_RUN_NAME}")_${TS}.log"
fi

if [[ "$MODE" == "tail" ]]; then
  if [[ -z "$LOG_FILE_OVERRIDE" ]]; then
    LOG_FILE="$(latest_log_for_prefix "$LOG_DIR" "$WANDB_RUN_NAME")"
  fi
  require_file "$LOG_FILE" "log file"
  echo "[INFO] Following log: $LOG_FILE"
  tail -n 100 -f "$LOG_FILE"
  exit 0
fi

if [[ "$MODE" == "nohup" || "$MODE" == "detach" ]]; then
  mkdir -p "$(dirname -- "$LOG_FILE")"
  touch "$LOG_FILE"
  CHILD_ARGS=(--run --from-nohup --log-file "$LOG_FILE")
  if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    CHILD_ARGS+=("${FORWARD_ARGS[@]}")
  fi
  nohup bash "$SCRIPT_PATH" "${CHILD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[INFO] RL started in background. pid=$PID"
  echo "[INFO] Log file: $LOG_FILE"
  if [[ "$MODE" == "nohup" ]]; then
    echo "[INFO] Press Ctrl-C to stop following logs (training keeps running)."
    if tail --help 2>&1 | grep -q -- '--pid'; then
      tail --pid="$PID" -n 100 -f "$LOG_FILE"
    else
      tail -n 100 -f "$LOG_FILE"
    fi
  fi
  exit 0
fi

if [[ ! -f "$CONDA_ACTIVATE" ]]; then
  echo "[ERROR] Conda activate script not found: $CONDA_ACTIVATE"
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"

export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset HF_ENDPOINT || true
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[INFO] Dry-run mode enabled."
fi

require_exists "$REPO_ROOT" "REPO_ROOT"
cd "$REPO_ROOT"

require_exists "$MODEL_PATH" "model path"
require_file "${DATA_DIR}/train.json" "RL train dataset"
require_file "${DATA_DIR}/valid.json" "RL valid dataset"
require_file "${DATA_DIR}/test.json" "RL test dataset"
require_file "$INDEX_PATH" "id2sid index file"
require_file "$DS_CONFIG" "DeepSpeed config"
require_file "${REPO_ROOT}/trl_trainer.py" "trl_trainer.py"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] accelerate not found in PATH"
  exit 1
fi

TRAIN_CMD=(
  accelerate launch
  --config_file "$DS_CONFIG"
  --num_processes "$NUM_PROCESSES"
  --main_process_port "$MAIN_PORT"
  trl_trainer.py
  --model "$MODEL_PATH"
  --data_dir "$DATA_DIR"
  --index_path "$INDEX_PATH"
  --output_dir "$OUTPUT_DIR"
  --num_beams "$NUM_BEAMS"
  --sid_levels "$SID_LEVELS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BSZ"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BSZ"
  --gradient_accumulation_steps "$GRAD_ACC"
  --num_train_epochs "$NUM_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --eval_step "$EVAL_STEP"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --beta "$BETA"
  --temperature "$TEMPERATURE"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --report_to "$REPORT_TO"
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
  exit 0
fi

mkdir -p "$(dirname -- "$LOG_FILE")"
if [[ "$FROM_NOHUP" -eq 0 ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "[INFO] CATEGORY=$CATEGORY"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] INDEX_PATH=$INDEX_PATH"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] DS_CONFIG=$DS_CONFIG"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] NUM_PROCESSES=$NUM_PROCESSES"
echo "[INFO] MAIN_PORT=$MAIN_PORT"
echo "[INFO] RUN_NAME=$WANDB_RUN_NAME"
echo "[INFO] RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"
echo "[INFO] SID_LEVELS=$SID_LEVELS"

"${TRAIN_CMD[@]}"
