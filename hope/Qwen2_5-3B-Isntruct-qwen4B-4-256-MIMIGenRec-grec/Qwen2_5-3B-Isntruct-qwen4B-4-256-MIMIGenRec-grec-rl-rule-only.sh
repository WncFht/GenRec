#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only.sh [overrides]

Overrides:
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
  --eval-on-start <true|false>
  --max-completion-length <n>
  --beta <float>
  --temperature <float>
  --save-total-limit <n>
  --report-to <name>
  --wandb-mode <offline|online|disabled>
  --conda-activate <path>
  --conda-env <name>
  --dry-run
  -h, --help
EOF
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

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-genrec}"
REPO_ROOT="${REPO_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec}"

DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/${DATA_VARIANT_DEFAULT}/rl}"
INDEX_PATH="${INDEX_PATH:-${REPO_ROOT}/data/${DATA_VARIANT_DEFAULT}/id2sid.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495}"
DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/config/zero2.yaml}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PORT="${MAIN_PORT:-29516}"
NUM_BEAMS="${NUM_BEAMS:-16}"
SID_LEVELS="${SID_LEVELS:--1}"
PER_DEVICE_TRAIN_BSZ="${PER_DEVICE_TRAIN_BSZ:-64}"
PER_DEVICE_EVAL_BSZ="${PER_DEVICE_EVAL_BSZ:-64}"
GRAD_ACC="${GRAD_ACC:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EVAL_STEP="${EVAL_STEP:-100}"
EVAL_ON_START="${EVAL_ON_START:-true}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
BETA="${BETA:-1e-3}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
REPORT_TO="${REPORT_TO:-wandb}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
RUN_NAME="${RUN_NAME:-${WANDB_RUN_NAME:-instruments_grec_rl_rule_only_rerun_quietlog_qwen2_5_3b_qwen4b_4_256_from_ckpt495}}"
DRY_RUN="${DRY_RUN:-0}"

export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --index-path)
      INDEX_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --resume|--resume-from-checkpoint)
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --port)
      MAIN_PORT="$2"
      shift 2
      ;;
    --ds-config)
      DS_CONFIG="$2"
      shift 2
      ;;
    --num-beams)
      NUM_BEAMS="$2"
      shift 2
      ;;
    --sid-levels)
      SID_LEVELS="$2"
      shift 2
      ;;
    --train-bsz)
      PER_DEVICE_TRAIN_BSZ="$2"
      shift 2
      ;;
    --eval-bsz)
      PER_DEVICE_EVAL_BSZ="$2"
      shift 2
      ;;
    --grad-acc)
      GRAD_ACC="$2"
      shift 2
      ;;
    --epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --eval-step)
      EVAL_STEP="$2"
      shift 2
      ;;
    --eval-on-start|--eval_on_start)
      EVAL_ON_START="$2"
      shift 2
      ;;
    --max-completion-length)
      MAX_COMPLETION_LENGTH="$2"
      shift 2
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --save-total-limit)
      SAVE_TOTAL_LIMIT="$2"
      shift 2
      ;;
    --report-to)
      REPORT_TO="$2"
      shift 2
      ;;
    --wandb-mode)
      export WANDB_MODE="$2"
      shift 2
      ;;
    --conda-activate)
      CONDA_ACTIVATE="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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

export WANDB_RUN_NAME="$RUN_NAME"

LAUNCH_ARGS=(
  accelerate launch
  --config_file "$DS_CONFIG"
  --num_processes "$NUM_PROCESSES"
  --main_process_port "$MAIN_PORT"
)

MODEL_ARGS=(
  trl_trainer.py
  --model "$MODEL_PATH"
  --data_dir "$DATA_DIR"
  --index_path "$INDEX_PATH"
  --output_dir "$OUTPUT_DIR"
)

GENERATION_ARGS=(
  --num_beams "$NUM_BEAMS"
  --sid_levels "$SID_LEVELS"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --temperature "$TEMPERATURE"
)

TRAINING_ARGS=(
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BSZ"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BSZ"
  --gradient_accumulation_steps "$GRAD_ACC"
  --num_train_epochs "$NUM_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --eval_step "$EVAL_STEP"
  --eval_on_start "$EVAL_ON_START"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --save_only_model true
)

REWARD_ARGS=(
  --beta "$BETA"
  --reward_mode rule_only
  --prefix_reward_normalize true
  --probe_rule_with_zero_weight false
  --token_level_prefix_advantage false
)

RUNTIME_ARGS=(
  --report_to "$REPORT_TO"
  --run_name "$RUN_NAME"
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
)

TRAIN_CMD=(
  "${LAUNCH_ARGS[@]}"
  "${MODEL_ARGS[@]}"
  "${GENERATION_ARGS[@]}"
  "${TRAINING_ARGS[@]}"
  "${REWARD_ARGS[@]}"
  "${RUNTIME_ARGS[@]}"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
  exit 0
fi

require_exists "$REPO_ROOT" "REPO_ROOT"
require_exists "$MODEL_PATH" "model path"
require_file "${DATA_DIR}/train.json" "RL train dataset"
require_file "${DATA_DIR}/valid.json" "RL valid dataset"
require_file "${DATA_DIR}/test.json" "RL test dataset"
require_file "$INDEX_PATH" "id2sid index file"
require_file "$DS_CONFIG" "DeepSpeed config"
require_file "${REPO_ROOT}/trl_trainer.py" "trl_trainer.py"

if [[ ! -f "$CONDA_ACTIVATE" ]]; then
  echo "[ERROR] Conda activate script not found: $CONDA_ACTIVATE"
  exit 1
fi

# shellcheck disable=SC1090
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
  source "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
fi
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
fi

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
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] accelerate not found in PATH"
  exit 1
fi

cd "$REPO_ROOT"
set -x
"${TRAIN_CMD[@]}"
