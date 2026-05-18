#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash hope/_canonical_rl_launcher.sh [--ds-config <path>] [--dry-run]

This launcher is configured through environment variables by thin wrappers under
hope/*-genrec and hope/*-lcrec.
EOF
}

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
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

require_transformers_payload() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing model directory: $path"
    exit 1
  fi
  if [[ ! -f "$path/config.json" ]]; then
    echo "[ERROR] Missing config.json under resolved model directory: $path"
    exit 1
  fi
  if [[ ! -f "$path/tokenizer.json" && ! -f "$path/tokenizer_config.json" && ! -f "$path/vocab.txt" ]]; then
    echo "[ERROR] Missing tokenizer files under resolved model directory: $path"
    exit 1
  fi
}

path_exists() {
  local path="$1"
  [[ -n "$path" && -e "$path" ]]
}

resolve_model_payload_dir() {
  local path="$1"
  local checkpoint_best="$path/checkpoint-best"
  local latest_checkpoint=""

  if [[ -d "$path" && -f "$path/config.json" ]]; then
    echo "$path"
    return 0
  fi
  if [[ -d "$checkpoint_best" && -f "$checkpoint_best/config.json" ]]; then
    echo "$checkpoint_best"
    return 0
  fi
  if [[ -d "$path" ]]; then
    latest_checkpoint="$(
      find "$path" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
    )"
    if [[ -n "$latest_checkpoint" && -f "$latest_checkpoint/config.json" ]]; then
      echo "$latest_checkpoint"
      return 0
    fi
  fi
  echo "$path"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_launcher_runtime.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_fixed_hint_artifacts.sh"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-genrec}"
REPO_ROOT="${REPO_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-}"
MODEL_PATH="${MODEL_PATH:-}"
DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/config/zero3.yaml}"
DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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
RUN_NAME="${RUN_NAME:-}"
REWARD_MODE="${REWARD_MODE:-rule_only}"

FIXED_HINT_ENABLED="${FIXED_HINT_ENABLED:-false}"
FIXED_HINT_APPLY_TO_EVAL="${FIXED_HINT_APPLY_TO_EVAL:-false}"
HINT_CE_LOSS_COEF="${HINT_CE_LOSS_COEF:-0.0}"
FULL_SEQUENCE_SFT_LOSS_COEF="${FULL_SEQUENCE_SFT_LOSS_COEF:-0.0}"
FORCE_REANALYZE="${FORCE_REANALYZE:-false}"
BEAM_SIZE="${BEAM_SIZE:-16}"
UNSOLVED_DEPTH="${UNSOLVED_DEPTH:-3}"
CAP_DEPTH="${CAP_DEPTH:-}"
ANALYZE_HINT_DEPTH="${ANALYZE_HINT_DEPTH:-1}"
ANALYZE_MAX_HINT_DEPTH="${ANALYZE_MAX_HINT_DEPTH:-3}"
ANALYZE_BATCH_SIZE="${ANALYZE_BATCH_SIZE:-8}"
ANALYZE_MAX_PROMPT_LENGTH="${ANALYZE_MAX_PROMPT_LENGTH:-512}"
ANALYZE_MAX_NEW_TOKENS="${ANALYZE_MAX_NEW_TOKENS:-128}"
ANALYZE_REPETITION_PENALTY="${ANALYZE_REPETITION_PENALTY:-1.0}"
ANALYSIS_DIR_DEFAULT="${ANALYSIS_DIR_DEFAULT:-${REPO_ROOT}/temp/rl_beam_hint/artifacts}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ds-config)
      DS_CONFIG="$2"
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

if [[ -z "$DATA_VARIANT_DEFAULT" ]]; then
  echo "[ERROR] DATA_VARIANT_DEFAULT is required"
  exit 1
fi
if [[ -z "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH is required"
  exit 1
fi

DATA_VARIANT_DIR="$(resolve_data_variant_dir "$REPO_ROOT" "$DATA_VARIANT_DEFAULT")"
DATA_DIR="${DATA_DIR:-${DATA_VARIANT_DIR}/rl}"
INDEX_PATH="${INDEX_PATH:-${DATA_VARIANT_DIR}/id2sid.json}"
ADD_TOKENS_PATH="${ADD_TOKENS_PATH:-${DATA_VARIANT_DIR}/new_tokens.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/$(basename "${DATA_VARIANT_DIR}")-rl}"
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(basename "${OUTPUT_DIR}" | tr '-' '_')"
fi

RESOLVED_MODEL_PATH="$(resolve_model_payload_dir "$MODEL_PATH")"
ANALYSIS_DATASET_ID="${ANALYSIS_DATASET_ID:-$(default_fixed_hint_dataset_id "$DATA_DIR")}"
ANALYSIS_SCOPE_ID="${ANALYSIS_SCOPE_ID:-all}"
ANALYSIS_MODEL_ID="${ANALYSIS_MODEL_ID:-$(default_fixed_hint_model_id "$RESOLVED_MODEL_PATH")}"

init_fixed_hint_artifact_paths \
  "$ANALYSIS_DIR_DEFAULT" \
  "$ANALYSIS_DATASET_ID" \
  "$ANALYSIS_SCOPE_ID" \
  "$ANALYSIS_MODEL_ID" \
  "$BEAM_SIZE" \
  "$ANALYZE_MAX_HINT_DEPTH" \
  "$SID_LEVELS" \
  "$UNSOLVED_DEPTH"

export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
export WANDB_RUN_NAME="$RUN_NAME"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"

LAUNCH_ARGS=(
  accelerate launch
  --config_file "$DS_CONFIG"
  --num_processes "$NUM_PROCESSES"
  --main_process_port "$MAIN_PORT"
)

MODEL_ARGS=(
  trl_trainer.py
  --model "$RESOLVED_MODEL_PATH"
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
  --reward_mode "$REWARD_MODE"
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
)

ANALYZE_CMD=()
ANALYSIS_STATUS="disabled"
ANALYSIS_HAVE_SUMMARY=0
ANALYSIS_HAVE_DETAILS=0
ANALYSIS_HAVE_MAP=0
if is_true "$FIXED_HINT_ENABLED"; then
  if path_exists "$ANALYSIS_SUMMARY_PATH"; then
    ANALYSIS_HAVE_SUMMARY=1
  fi
  if path_exists "$ANALYSIS_DETAILS_PATH"; then
    ANALYSIS_HAVE_DETAILS=1
  fi
  if path_exists "$FIXED_HINT_MAP_PATH"; then
    ANALYSIS_HAVE_MAP=1
  fi

  if ! is_true "$FORCE_REANALYZE" && [[ "$ANALYSIS_HAVE_SUMMARY" == "1" && "$ANALYSIS_HAVE_DETAILS" == "1" && "$ANALYSIS_HAVE_MAP" == "1" ]]; then
    ANALYSIS_STATUS="skip-existing"
  else
    ANALYSIS_STATUS="run"
  fi

  ANALYZE_CMD=(
    "$PYTHON_BIN"
    analyze_rl_beam_hint.py
    --model-path "$RESOLVED_MODEL_PATH"
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
    --export-fixed-hint-depth-map-path "$FIXED_HINT_MAP_PATH"
    --export-fixed-hint-beam-size "$BEAM_SIZE"
    --export-fixed-hint-unsolved-depth "$UNSOLVED_DEPTH"
  )
  if [[ "$ANALYSIS_HAVE_SUMMARY" == "1" ]]; then
    ANALYZE_CMD+=(--reuse-summary-path "$ANALYSIS_SUMMARY_PATH")
  fi
  if [[ "$ANALYSIS_HAVE_DETAILS" == "1" ]]; then
    ANALYZE_CMD+=(--reuse-details-path "$ANALYSIS_DETAILS_PATH")
  fi
  FIXED_HINT_ARGS=(
    --fixed_hint_depth_map_path "$FIXED_HINT_MAP_PATH"
    --fixed_hint_unsolved_depth "$UNSOLVED_DEPTH"
    --fixed_hint_apply_to_eval "$FIXED_HINT_APPLY_TO_EVAL"
    --hint_ce_loss_coef "$HINT_CE_LOSS_COEF"
    --full_sequence_sft_loss_coef "$FULL_SEQUENCE_SFT_LOSS_COEF"
  )
  TRAIN_CMD+=("${FIXED_HINT_ARGS[@]}")
  if [[ -n "$CAP_DEPTH" ]]; then
    TRAIN_CMD+=(--fixed_hint_depth_cap "$CAP_DEPTH")
  fi
fi

TRAIN_CMD+=("${RUNTIME_ARGS[@]}")

if [[ "$DRY_RUN" == "1" ]]; then
  if [[ "${#ANALYZE_CMD[@]}" -gt 0 && "$ANALYSIS_STATUS" == "run" ]]; then
    printf '%q ' "${ANALYZE_CMD[@]}"
    echo
  fi
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
  exit 0
fi

require_exists "$REPO_ROOT" "REPO_ROOT"
require_file "$CONDA_ACTIVATE" "conda activate script"
require_exists "$MODEL_PATH" "MODEL_PATH"
require_transformers_payload "$RESOLVED_MODEL_PATH"
require_file "${DATA_DIR}/train.json" "RL train dataset"
require_file "${DATA_DIR}/valid.json" "RL valid dataset"
require_file "${DATA_DIR}/test.json" "RL test dataset"
require_file "$INDEX_PATH" "id2sid index file"
require_file "$DS_CONFIG" "DeepSpeed config"
require_file "${REPO_ROOT}/trl_trainer.py" "trl_trainer.py"

activate_genrec_env "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
setup_genrec_runtime_env "$REPO_ROOT"
require_accelerate

if [[ "${#ANALYZE_CMD[@]}" -gt 0 ]]; then
  require_file "$ADD_TOKENS_PATH" "new tokens file"
  require_file "${REPO_ROOT}/analyze_rl_beam_hint.py" "analyze_rl_beam_hint.py"
  mkdir -p "$(dirname -- "$ANALYSIS_SUMMARY_PATH")"
  mkdir -p "$(dirname -- "$ANALYSIS_DETAILS_PATH")"
  mkdir -p "$(dirname -- "$FIXED_HINT_MAP_PATH")"
fi

cd "$REPO_ROOT"

echo "[INFO] DATA_VARIANT_DEFAULT=${DATA_VARIANT_DEFAULT}"
echo "[INFO] DATA_VARIANT_DIR=${DATA_VARIANT_DIR}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] RESOLVED_MODEL_PATH=${RESOLVED_MODEL_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] ANALYSIS_STATUS=${ANALYSIS_STATUS}"
if is_true "$FIXED_HINT_ENABLED"; then
  echo "[INFO] ANALYSIS_HAVE_SUMMARY=${ANALYSIS_HAVE_SUMMARY}"
  echo "[INFO] ANALYSIS_HAVE_DETAILS=${ANALYSIS_HAVE_DETAILS}"
  echo "[INFO] ANALYSIS_HAVE_MAP=${ANALYSIS_HAVE_MAP}"
  echo "[INFO] FORCE_REANALYZE=${FORCE_REANALYZE}"
fi

if [[ "${#ANALYZE_CMD[@]}" -gt 0 && "$ANALYSIS_STATUS" == "run" ]]; then
  set -x
  "${ANALYZE_CMD[@]}"
  set +x

  require_file "$ANALYSIS_SUMMARY_PATH" "analysis summary"
  require_file "$ANALYSIS_DETAILS_PATH" "analysis details"
  require_file "$FIXED_HINT_MAP_PATH" "fixed hint map"
elif is_true "$FIXED_HINT_ENABLED"; then
  require_file "$ANALYSIS_SUMMARY_PATH" "analysis summary"
  require_file "$ANALYSIS_DETAILS_PATH" "analysis details"
  require_file "$FIXED_HINT_MAP_PATH" "fixed hint map"
fi

set -x
"${TRAIN_CMD[@]}"
