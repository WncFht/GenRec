set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash trl_trainer.sh [options]

Modes:
  --nohup               Run in background with nohup and follow log (Ctrl-C only stops follow)
  --detach              Run in background with nohup and do not follow log
  --tail                Follow the current run log and exit

Config overrides:
  --category <name>
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

MODE="foreground" # foreground | nohup | detach | tail | run(internal)
DRY_RUN=0

export CATEGORY="Industrial_and_Scientific"
USER_SET_MODEL_PATH=0
USER_SET_DATA_DIR=0
USER_SET_INDEX_PATH=0
USER_SET_OUTPUT_DIR=0
USER_SET_RUN_NAME=0

MODEL_PATH=""
DATA_DIR=""
INDEX_PATH=""
OUTPUT_DIR=""

# offline mode (use local checkpoints/datasets only)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset HF_ENDPOINT || true

# wandb defaults
export WANDB_PROJECT="MIMIGenRec-GRPO"
export WANDB_MODE="offline"
export WANDB_API_KEY=""
REPORT_TO="wandb"

# training defaults
NUM_PROCESSES=4
MAIN_PORT=29503
DS_CONFIG="config/zero2.yaml"
NUM_BEAMS=16
PER_DEVICE_TRAIN_BSZ=64
PER_DEVICE_EVAL_BSZ=64
GRAD_ACC=4
NUM_EPOCHS=2
LEARNING_RATE="1e-5"
EVAL_STEP=20
MAX_COMPLETION_LENGTH=128
BETA="1e-3"
TEMPERATURE="1.0"
SAVE_TOTAL_LIMIT=1
RESUME_FROM_CHECKPOINT="auto"

# 4-GPU default
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

FORWARD_ARGS=()
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
    --category)
      CATEGORY="$2"
      FORWARD_ARGS+=("--category" "$2")
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      USER_SET_MODEL_PATH=1
      FORWARD_ARGS+=("--model-path" "$2")
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      USER_SET_DATA_DIR=1
      FORWARD_ARGS+=("--data-dir" "$2")
      shift 2
      ;;
    --index-path)
      INDEX_PATH="$2"
      USER_SET_INDEX_PATH=1
      FORWARD_ARGS+=("--index-path" "$2")
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      USER_SET_OUTPUT_DIR=1
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
      USER_SET_RUN_NAME=1
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

if [[ $USER_SET_MODEL_PATH -eq 0 ]]; then
  MODEL_PATH="saves/qwen2.5-3b/full/${CATEGORY}-sft-dsz0-4gpu-eq8/checkpoint-260"
fi
if [[ $USER_SET_DATA_DIR -eq 0 ]]; then
  DATA_DIR="data/${CATEGORY}/rl"
fi
if [[ $USER_SET_INDEX_PATH -eq 0 ]]; then
  INDEX_PATH="data/${CATEGORY}/id2sid.json"
fi
if [[ $USER_SET_OUTPUT_DIR -eq 0 ]]; then
  OUTPUT_DIR="rl_outputs/${CATEGORY}-qwen2.5-3b-instruct-grpo"
fi
if [[ $USER_SET_RUN_NAME -eq 0 ]]; then
  export WANDB_RUN_NAME="${CATEGORY}-qwen2.5-3b-instruct"
fi

LOG_DIR="log"
LOG_FILE="${LOG_DIR}/$(sanitize_log_name "${WANDB_RUN_NAME}").log"

if [[ "$MODE" == "tail" ]]; then
  require_file "$LOG_FILE" "log file"
  tail -n 100 -f "$LOG_FILE"
  exit 0
fi

if [[ "$MODE" == "nohup" || "$MODE" == "detach" ]]; then
  mkdir -p "$LOG_DIR"
  touch "$LOG_FILE"
  CHILD_ARGS=(--run)
  if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    CHILD_ARGS+=("${FORWARD_ARGS[@]}")
  fi
  nohup bash "$0" "${CHILD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[INFO] RL started in background. pid=$PID"
  echo "[INFO] Log file: $LOG_FILE"
  if [[ "$MODE" == "nohup" ]]; then
    echo "[INFO] Press Ctrl-C to stop following logs (training will continue)."
    if tail --help 2>&1 | grep -q -- '--pid'; then
      tail --pid="$PID" -n 100 -f "$LOG_FILE"
    else
      tail -n 100 -f "$LOG_FILE"
    fi
  fi
  exit 0
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
  echo "[INFO] Dry-run mode. Command:"
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
  exit 0
fi

require_exists "$MODEL_PATH" "model path"
require_file "${DATA_DIR}/train.json" "RL train dataset"
require_file "${DATA_DIR}/valid.json" "RL valid dataset"
require_file "$INDEX_PATH" "id2sid index file"
require_file "$DS_CONFIG" "DeepSpeed config"

if [[ "$MODE" == "foreground" ]]; then
  mkdir -p "$LOG_DIR"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] CATEGORY=$CATEGORY"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] NUM_PROCESSES=$NUM_PROCESSES"
echo "[INFO] MAIN_PORT=$MAIN_PORT"
echo "[INFO] RUN_NAME=$WANDB_RUN_NAME"
echo "[INFO] RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"

"${TRAIN_CMD[@]}"
