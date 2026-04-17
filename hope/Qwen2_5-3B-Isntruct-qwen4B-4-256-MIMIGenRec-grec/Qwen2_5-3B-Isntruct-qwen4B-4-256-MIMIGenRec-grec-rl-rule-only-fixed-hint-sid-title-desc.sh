#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-title-desc.sh [options]

Run modes:
  --nohup                 Generate dual-task hint analysis if needed and start RL in background via nohup (default)
  --detach                Start in background without log follow
  --tail                  Follow latest log for current run name
  --run                   Internal mode; execute analysis/export + train directly

Common overrides:
  --model-path <path>
  --data-dir <path>
  --index-path <path>
  --add-tokens-path <path>
  --output-dir <path>
  --resume <auto|none|path>
  --num-processes <n>
  --port <n>
  --ds-config <path>
  --num-beams <n>         Default: 16
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
  --report-to <name>
  --wandb-mode <offline|online|disabled>
  --analysis-summary-path <path>
  --analysis-details-path <path>
  --fixed-hint-map-path <path>
  --beam-size <n>         Default: 16
  --unsolved-depth <n>    Default: 3
  --cap <n>               Optional training-time cap over oracle depth
  --train-task-names <csv>     Default: task1_sid_sft,task5_title_desc2sid
  --eval-task-names <csv>      Default: task1_sid_sft
  --analysis-task-names <csv>  Default: task1_sid_sft,task5_title_desc2sid
  --run-name <name>
  --python-bin <python>
  --conda-activate <path>
  --conda-env <name>
  --log-dir <path>
  --log-file <path>
  --dry-run
  -h, --help
EOF
}

sanitize_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  echo "$value"
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
  latest="$(ls -1t "${log_dir}/$(sanitize_name "${prefix}")"_*.log 2>/dev/null | head -n 1 || true)"
  echo "$latest"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODE="nohup"
DRY_RUN=0
FROM_NOHUP=0
LOG_FILE_OVERRIDE=""

TS="$(date +%Y%m%d_%H%M%S)"
CONDA_ACTIVATE_DEFAULT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-$CONDA_ACTIVATE_DEFAULT}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-genrec}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/${DATA_VARIANT_DEFAULT}/rl}"
INDEX_PATH="${INDEX_PATH:-${REPO_ROOT}/data/${DATA_VARIANT_DEFAULT}/id2sid.json}"
ADD_TOKENS_PATH="${ADD_TOKENS_PATH:-${REPO_ROOT}/data/${DATA_VARIANT_DEFAULT}/new_tokens.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-title-desc-sft495}"
DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/config/zero2.yaml}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PORT="${MAIN_PORT:-29519}"
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
REPORT_TO="${REPORT_TO:-wandb}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"

RUN_NAME="${RUN_NAME:-instruments_grec_rl_rule_only_fixed_hint_taskfix_b16_sid_title_desc_ckpt495}"
ANALYSIS_RUN_NAME="${ANALYSIS_RUN_NAME:-instruments_grec_rlsid_title_desc_beam_hint_qwen2_5_3b_qwen4b_4_256_ckpt495}"
ANALYSIS_DIR_DEFAULT="${REPO_ROOT}/temp/rl_beam_hint"
ANALYSIS_SUMMARY_PATH="${ANALYSIS_SUMMARY_PATH:-${ANALYSIS_DIR_DEFAULT}/instruments_grec_rlsid_title_desc_beam_hint_cascade_summary.json}"
ANALYSIS_DETAILS_PATH="${ANALYSIS_DETAILS_PATH:-${ANALYSIS_DIR_DEFAULT}/instruments_grec_rlsid_title_desc_beam_hint_cascade_details.json}"
FIXED_HINT_MAP_PATH="${FIXED_HINT_MAP_PATH:-${ANALYSIS_DIR_DEFAULT}/$(sanitize_name "${RUN_NAME}")_${TS}_beam16_hint_map.json}"

BEAM_SIZE="${BEAM_SIZE:-16}"
UNSOLVED_DEPTH="${UNSOLVED_DEPTH:-3}"
CAP_DEPTH="${CAP_DEPTH:-}"
ANALYZE_HINT_DEPTH="${ANALYZE_HINT_DEPTH:-1}"
ANALYZE_MAX_HINT_DEPTH="${ANALYZE_MAX_HINT_DEPTH:-3}"
ANALYZE_BATCH_SIZE="${ANALYZE_BATCH_SIZE:-8}"
ANALYZE_MAX_PROMPT_LENGTH="${ANALYZE_MAX_PROMPT_LENGTH:-512}"
ANALYZE_MAX_NEW_TOKENS="${ANALYZE_MAX_NEW_TOKENS:-128}"
ANALYZE_REPETITION_PENALTY="${ANALYZE_REPETITION_PENALTY:-1.0}"
TRAIN_TASK_NAMES="${TRAIN_TASK_NAMES:-task1_sid_sft,task5_title_desc2sid}"
EVAL_TASK_NAMES="${EVAL_TASK_NAMES:-task1_sid_sft}"
ANALYSIS_TASK_NAMES="${ANALYSIS_TASK_NAMES:-${TRAIN_TASK_NAMES}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log}"

export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_NAME="$RUN_NAME"

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
    --add-tokens-path)
      ADD_TOKENS_PATH="$2"
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
    --report-to)
      REPORT_TO="$2"
      shift 2
      ;;
    --wandb-mode)
      export WANDB_MODE="$2"
      shift 2
      ;;
    --analysis-summary-path)
      ANALYSIS_SUMMARY_PATH="$2"
      shift 2
      ;;
    --analysis-details-path)
      ANALYSIS_DETAILS_PATH="$2"
      shift 2
      ;;
    --fixed-hint-map-path)
      FIXED_HINT_MAP_PATH="$2"
      shift 2
      ;;
    --beam-size)
      BEAM_SIZE="$2"
      shift 2
      ;;
    --unsolved-depth)
      UNSOLVED_DEPTH="$2"
      shift 2
      ;;
    --cap)
      CAP_DEPTH="$2"
      shift 2
      ;;
    --train-task-names|--train_task_names)
      TRAIN_TASK_NAMES="$2"
      shift 2
      ;;
    --eval-task-names|--eval_task_names)
      EVAL_TASK_NAMES="$2"
      shift 2
      ;;
    --analysis-task-names|--analysis_task_names)
      ANALYSIS_TASK_NAMES="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      export WANDB_RUN_NAME="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
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
  LOG_FILE="${LOG_DIR}/$(sanitize_name "${RUN_NAME}")_${TS}.log"
fi

if [[ "$MODE" == "tail" ]]; then
  if [[ -z "$LOG_FILE_OVERRIDE" ]]; then
    LOG_FILE="$(latest_log_for_prefix "$LOG_DIR" "$RUN_NAME")"
  fi
  require_file "$LOG_FILE" "log file"
  echo "[INFO] Following log: $LOG_FILE"
  tail -n 100 -f "$LOG_FILE"
  exit 0
fi

if [[ "$MODE" == "nohup" || "$MODE" == "detach" ]]; then
  mkdir -p "$(dirname -- "$LOG_FILE")"
  touch "$LOG_FILE"
  CHILD_ARGS=(
    --run
    --from-nohup
    --log-file "$LOG_FILE"
    --run-name "$RUN_NAME"
    --python-bin "$PYTHON_BIN"
    --conda-activate "$CONDA_ACTIVATE"
    --conda-env "$CONDA_ENV_NAME"
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    CHILD_ARGS+=(--dry-run)
  fi
  CHILD_ARGS+=(
    --model-path "$MODEL_PATH"
    --data-dir "$DATA_DIR"
    --index-path "$INDEX_PATH"
    --add-tokens-path "$ADD_TOKENS_PATH"
    --output-dir "$OUTPUT_DIR"
    --resume "$RESUME_FROM_CHECKPOINT"
    --num-processes "$NUM_PROCESSES"
    --port "$MAIN_PORT"
    --ds-config "$DS_CONFIG"
    --num-beams "$NUM_BEAMS"
    --sid-levels "$SID_LEVELS"
    --train-bsz "$PER_DEVICE_TRAIN_BSZ"
    --eval-bsz "$PER_DEVICE_EVAL_BSZ"
    --grad-acc "$GRAD_ACC"
    --epochs "$NUM_EPOCHS"
    --lr "$LEARNING_RATE"
    --eval-step "$EVAL_STEP"
    --max-completion-length "$MAX_COMPLETION_LENGTH"
    --beta "$BETA"
    --temperature "$TEMPERATURE"
    --report-to "$REPORT_TO"
    --analysis-summary-path "$ANALYSIS_SUMMARY_PATH"
    --analysis-details-path "$ANALYSIS_DETAILS_PATH"
    --fixed-hint-map-path "$FIXED_HINT_MAP_PATH"
    --beam-size "$BEAM_SIZE"
    --unsolved-depth "$UNSOLVED_DEPTH"
    --train-task-names "$TRAIN_TASK_NAMES"
    --eval-task-names "$EVAL_TASK_NAMES"
    --analysis-task-names "$ANALYSIS_TASK_NAMES"
  )
  if [[ -n "$CAP_DEPTH" ]]; then
    CHILD_ARGS+=(--cap "$CAP_DEPTH")
  fi
  nohup bash "$SCRIPT_PATH" "${CHILD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[INFO] Fixed-hint sid+title_desc RL pipeline started in background. pid=$PID"
  echo "[INFO] Log file: $LOG_FILE"
  if [[ "$MODE" == "nohup" ]]; then
    echo "[INFO] Press Ctrl-C to stop following logs (job keeps running)."
    if tail --help 2>&1 | grep -q -- '--pid'; then
      tail --pid="$PID" -n 100 -f "$LOG_FILE"
    else
      tail -n 100 -f "$LOG_FILE"
    fi
  fi
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
require_file "${REPO_ROOT}/analyze_rl_beam_hint.py" "analyze_rl_beam_hint.py"

ANALYZE_CMD=(
  "$PYTHON_BIN"
  analyze_rl_beam_hint.py
  --model-path "$MODEL_PATH"
  --data-dir "$DATA_DIR"
  --index-path "$INDEX_PATH"
  --add-tokens-path "$ADD_TOKENS_PATH"
  --summary-path "$ANALYSIS_SUMMARY_PATH"
  --details-path "$ANALYSIS_DETAILS_PATH"
  --beam-sizes "$BEAM_SIZE"
  --hint-depth "$ANALYZE_HINT_DEPTH"
  --max-hint-depth "$ANALYZE_MAX_HINT_DEPTH"
  --batch-size "$ANALYZE_BATCH_SIZE"
  --max-prompt-length "$ANALYZE_MAX_PROMPT_LENGTH"
  --max-new-tokens "$ANALYZE_MAX_NEW_TOKENS"
  --repetition-penalty "$ANALYZE_REPETITION_PENALTY"
  --sid-levels "$SID_LEVELS"
  --cache-dir "$ANALYSIS_DIR_DEFAULT"
  --task-names "$ANALYSIS_TASK_NAMES"
)

EXPORT_CMD=(
  "$PYTHON_BIN"
  analyze_rl_beam_hint.py
  --model-path "$MODEL_PATH"
  --data-dir "$DATA_DIR"
  --index-path "$INDEX_PATH"
  --add-tokens-path "$ADD_TOKENS_PATH"
  --beam-sizes "$BEAM_SIZE"
  --reuse-summary-path "$ANALYSIS_SUMMARY_PATH"
  --reuse-details-path "$ANALYSIS_DETAILS_PATH"
  --task-names "$ANALYSIS_TASK_NAMES"
  --export-fixed-hint-depth-map-path "$FIXED_HINT_MAP_PATH"
  --export-fixed-hint-beam-size "$BEAM_SIZE"
  --export-fixed-hint-unsolved-depth "$UNSOLVED_DEPTH"
)

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
  --eval_on_start "$EVAL_ON_START"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --beta "$BETA"
  --temperature "$TEMPERATURE"
  --reward_mode rule_only
  --prefix_reward_normalize true
  --probe_rule_with_zero_weight false
  --token_level_prefix_advantage false
  --save_total_limit 10
  --save_only_model true
  --report_to "$REPORT_TO"
  --run_name "$RUN_NAME"
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
  --fixed_hint_depth_map_path "$FIXED_HINT_MAP_PATH"
  --fixed_hint_unsolved_depth "$UNSOLVED_DEPTH"
  --fixed_hint_apply_to_eval false
  --train_task_names "$TRAIN_TASK_NAMES"
  --eval_task_names "$EVAL_TASK_NAMES"
)

if [[ -n "$CAP_DEPTH" ]]; then
  TRAIN_CMD+=(--fixed_hint_depth_cap "$CAP_DEPTH")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ ! -f "$ANALYSIS_SUMMARY_PATH" || ! -f "$ANALYSIS_DETAILS_PATH" ]]; then
    printf '%q ' "${ANALYZE_CMD[@]}"
    echo
  fi
  printf '%q ' "${EXPORT_CMD[@]}"
  echo
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
  exit 0
fi

mkdir -p "$(dirname -- "$LOG_FILE")"
mkdir -p "$(dirname -- "$ANALYSIS_SUMMARY_PATH")"
mkdir -p "$(dirname -- "$ANALYSIS_DETAILS_PATH")"
mkdir -p "$(dirname -- "$FIXED_HINT_MAP_PATH")"
if [[ "$FROM_NOHUP" -eq 0 ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] INDEX_PATH=$INDEX_PATH"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] DS_CONFIG=$DS_CONFIG"
echo "[INFO] NUM_PROCESSES=$NUM_PROCESSES"
echo "[INFO] MAIN_PORT=$MAIN_PORT"
echo "[INFO] NUM_BEAMS=$NUM_BEAMS"
echo "[INFO] SID_LEVELS=$SID_LEVELS"
echo "[INFO] TRAIN_BSZ=$PER_DEVICE_TRAIN_BSZ"
echo "[INFO] EVAL_BSZ=$PER_DEVICE_EVAL_BSZ"
echo "[INFO] GRAD_ACC=$GRAD_ACC"
echo "[INFO] NUM_EPOCHS=$NUM_EPOCHS"
echo "[INFO] LEARNING_RATE=$LEARNING_RATE"
echo "[INFO] EVAL_STEP=$EVAL_STEP"
echo "[INFO] MAX_COMPLETION_LENGTH=$MAX_COMPLETION_LENGTH"
echo "[INFO] BETA=$BETA"
echo "[INFO] TEMPERATURE=$TEMPERATURE"
echo "[INFO] REWARD_MODE=rule_only"
echo "[INFO] PREFIX_REWARD_NORMALIZE=true"
echo "[INFO] PROBE_RULE_ZERO_WEIGHT=false"
echo "[INFO] TOKEN_LEVEL_PREFIX_ADV=false"
echo "[INFO] REPORT_TO=$REPORT_TO"
echo "[INFO] RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"
echo "[INFO] ANALYSIS_RUN_NAME=$ANALYSIS_RUN_NAME"
echo "[INFO] ANALYSIS_SUMMARY_PATH=$ANALYSIS_SUMMARY_PATH"
echo "[INFO] ANALYSIS_DETAILS_PATH=$ANALYSIS_DETAILS_PATH"
echo "[INFO] FIXED_HINT_MAP_PATH=$FIXED_HINT_MAP_PATH"
echo "[INFO] FIXED_HINT_GENERATION_MODE=mixed_single_generate"
echo "[INFO] BEAM_SIZE=$BEAM_SIZE"
echo "[INFO] UNSOLVED_DEPTH=$UNSOLVED_DEPTH"
echo "[INFO] CAP_DEPTH=${CAP_DEPTH:-<none>}"
echo "[INFO] TRAIN_TASK_NAMES=$TRAIN_TASK_NAMES"
echo "[INFO] EVAL_TASK_NAMES=$EVAL_TASK_NAMES"
echo "[INFO] ANALYSIS_TASK_NAMES=$ANALYSIS_TASK_NAMES"
echo "[INFO] LOG_FILE=$LOG_FILE"

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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
unset HF_ENDPOINT || true
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "$REPO_ROOT"

if [[ ! -f "$ANALYSIS_SUMMARY_PATH" || ! -f "$ANALYSIS_DETAILS_PATH" ]]; then
  echo "[INFO] dual-task hint analysis cache missing, generating summary/details first."
  "${ANALYZE_CMD[@]}"
fi

require_file "$ANALYSIS_SUMMARY_PATH" "analysis summary"
require_file "$ANALYSIS_DETAILS_PATH" "analysis details"

"${EXPORT_CMD[@]}"
"${TRAIN_CMD[@]}"
