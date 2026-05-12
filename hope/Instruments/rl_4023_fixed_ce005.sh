#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash rl_4023_fixed_ce005.sh [--ds-config <path>] [--dry-run]

  --ds-config <path>
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

CONDA_ACTIVATE="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate"
CONDA_ENV_NAME="genrec"
REPO_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec"
PYTHON_BIN="python"

# shellcheck disable=SC1091
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_fixed_hint_artifacts.sh"
# shellcheck disable=SC1091
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/_launcher_runtime.sh"

DATA_VARIANT_DEFAULT="Instruments_grec_index_lcrec"
DATA_VARIANT_DIR="$(resolve_data_variant_dir "$REPO_ROOT" "$DATA_VARIANT_DEFAULT")"
MODEL_PATH="${REPO_ROOT}/saves/qwen2.5-3b/full/Instruments-grec-lcrec-aligned-sft-qwen4B-4-256-dsz3-4gpu/checkpoint-4023"
DATA_DIR="${DATA_VARIANT_DIR}/rl"
INDEX_PATH="${DATA_VARIANT_DIR}/id2sid.json"
ADD_TOKENS_PATH="${DATA_VARIANT_DIR}/new_tokens.json"
OUTPUT_DIR="${REPO_ROOT}/rl_outputs/Instruments-grec-lc4023-fixed-ce005"
DS_CONFIG="${REPO_ROOT}/config/zero2.yaml"

NUM_PROCESSES=4
MAIN_PORT=29516
NUM_BEAMS=16
SID_LEVELS=-1
PER_DEVICE_TRAIN_BSZ=64
PER_DEVICE_EVAL_BSZ=64
GRAD_ACC=4
NUM_EPOCHS=2
LEARNING_RATE=1e-5
EVAL_STEP=100
EVAL_ON_START=true
MAX_COMPLETION_LENGTH=128
BETA=1e-3
TEMPERATURE=1.0
SAVE_TOTAL_LIMIT=10
REPORT_TO="wandb"
RESUME_FROM_CHECKPOINT="auto"

RUN_NAME="instruments_grec_lc4023_fixed_ce005"
ANALYSIS_DIR_DEFAULT="${REPO_ROOT}/temp/rl_beam_hint/artifacts"
ANALYSIS_DATASET_ID="$(default_fixed_hint_dataset_id "$DATA_DIR")"
ANALYSIS_TASK_NAMES=""
ANALYSIS_SCOPE_ID="all"
ANALYSIS_MODEL_ID="$(default_fixed_hint_model_id "$MODEL_PATH")"

BEAM_SIZE=16
UNSOLVED_DEPTH=3
CAP_DEPTH=""
HINT_CE_LOSS_COEF=0.005
ANALYZE_HINT_DEPTH=1
ANALYZE_MAX_HINT_DEPTH=3
ANALYZE_BATCH_SIZE=8
ANALYZE_MAX_PROMPT_LENGTH=512
ANALYZE_MAX_NEW_TOKENS=128
ANALYZE_REPETITION_PENALTY=1.0
DRY_RUN=0

init_fixed_hint_artifact_paths \
  "$ANALYSIS_DIR_DEFAULT" \
  "$ANALYSIS_DATASET_ID" \
  "$ANALYSIS_SCOPE_ID" \
  "$ANALYSIS_MODEL_ID" \
  "$BEAM_SIZE" \
  "$ANALYZE_MAX_HINT_DEPTH" \
  "$SID_LEVELS" \
  "$UNSOLVED_DEPTH"

export WANDB_PROJECT="MIMIGenRec-GRPO"
export WANDB_MODE="offline"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

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

export WANDB_RUN_NAME="$RUN_NAME"

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
  --export-fixed-hint-depth-map-path "$FIXED_HINT_MAP_PATH"
  --export-fixed-hint-beam-size "$BEAM_SIZE"
  --export-fixed-hint-unsolved-depth "$UNSOLVED_DEPTH"
)

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

FIXED_HINT_ARGS=(
  --fixed_hint_depth_map_path "$FIXED_HINT_MAP_PATH"
  --fixed_hint_unsolved_depth "$UNSOLVED_DEPTH"
  --fixed_hint_apply_to_eval false
  --hint_ce_loss_coef "$HINT_CE_LOSS_COEF"
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
  "${FIXED_HINT_ARGS[@]}"
  "${RUNTIME_ARGS[@]}"
)

if [[ -n "$CAP_DEPTH" ]]; then
  TRAIN_CMD+=(--fixed_hint_depth_cap "$CAP_DEPTH")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${ANALYZE_CMD[@]}"
  echo
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
require_file "$ADD_TOKENS_PATH" "new tokens file"
require_file "$DS_CONFIG" "DeepSpeed config"
require_file "${REPO_ROOT}/trl_trainer.py" "trl_trainer.py"
require_file "${REPO_ROOT}/analyze_rl_beam_hint.py" "analyze_rl_beam_hint.py"

activate_genrec_env "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
setup_genrec_runtime_env "$REPO_ROOT"
require_accelerate

mkdir -p "$(dirname -- "$ANALYSIS_SUMMARY_PATH")"
mkdir -p "$(dirname -- "$ANALYSIS_DETAILS_PATH")"
mkdir -p "$(dirname -- "$FIXED_HINT_MAP_PATH")"

cd "$REPO_ROOT"

set -x
"${ANALYZE_CMD[@]}"
set +x

require_file "$ANALYSIS_SUMMARY_PATH" "analysis summary"
require_file "$ANALYSIS_DETAILS_PATH" "analysis details"
require_file "$FIXED_HINT_MAP_PATH" "fixed hint map"

set -x
"${TRAIN_CMD[@]}"
