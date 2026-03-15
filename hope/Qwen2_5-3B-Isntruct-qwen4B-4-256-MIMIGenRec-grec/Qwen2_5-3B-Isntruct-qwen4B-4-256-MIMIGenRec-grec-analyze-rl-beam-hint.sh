#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-analyze-rl-beam-hint.sh [options]

Run modes:
  --nohup                 Start analysis in background via nohup and follow log (default)
  --detach                Start analysis in background via nohup and do not follow log
  --tail                  Follow latest log for current run name
  --run                   Internal mode; run analysis command directly

Common overrides:
  --model-path <path>
  --data-dir <path>
  --index-path <path>
  --add-tokens-path <path>
  --summary-path <path>
  --details-path <path>
  --run-name <name>
  --beam-sizes <csv>      Default: 8,16
  --hint-depth <n>        Default: 1
  --max-hint-depth <n>    Default: 3
  --batch-size <n>
  --max-samples <n>
  --offset <n>
  --max-prompt-length <n>
  --max-new-tokens <n>
  --repetition-penalty <float>
  --sid-levels <n>
  --cache-dir <path>
  --reuse-summary-path <path>
  --reuse-details-path <path>
  --disable-cache-reuse
  --trust-remote-code
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
TRUST_REMOTE_CODE=0
LOG_FILE_OVERRIDE=""
FORWARD_ARGS=()

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

RUN_NAME="${RUN_NAME:-instruments_grec_beam_hint_qwen2_5_3b_qwen4b_4_256_ckpt495}"
SUMMARY_DIR_DEFAULT="${REPO_ROOT}/temp/rl_beam_hint"
SUMMARY_PATH="${SUMMARY_PATH:-}"
DETAILS_PATH="${DETAILS_PATH:-}"

BEAM_SIZES="${BEAM_SIZES:-8,16}"
HINT_DEPTH="${HINT_DEPTH:-1}"
MAX_HINT_DEPTH="${MAX_HINT_DEPTH:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OFFSET="${OFFSET:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SID_LEVELS="${SID_LEVELS:--1}"
CACHE_DIR="${CACHE_DIR:-${SUMMARY_DIR_DEFAULT}}"
REUSE_SUMMARY_PATH="${REUSE_SUMMARY_PATH:-}"
REUSE_DETAILS_PATH="${REUSE_DETAILS_PATH:-}"
DISABLE_CACHE_REUSE=0

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
    --add-tokens-path)
      ADD_TOKENS_PATH="$2"
      FORWARD_ARGS+=("--add-tokens-path" "$2")
      shift 2
      ;;
    --summary-path)
      SUMMARY_PATH="$2"
      FORWARD_ARGS+=("--summary-path" "$2")
      shift 2
      ;;
    --details-path)
      DETAILS_PATH="$2"
      FORWARD_ARGS+=("--details-path" "$2")
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      FORWARD_ARGS+=("--run-name" "$2")
      shift 2
      ;;
    --beam-sizes)
      BEAM_SIZES="$2"
      FORWARD_ARGS+=("--beam-sizes" "$2")
      shift 2
      ;;
    --hint-depth)
      HINT_DEPTH="$2"
      FORWARD_ARGS+=("--hint-depth" "$2")
      shift 2
      ;;
    --max-hint-depth)
      MAX_HINT_DEPTH="$2"
      FORWARD_ARGS+=("--max-hint-depth" "$2")
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      FORWARD_ARGS+=("--batch-size" "$2")
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      FORWARD_ARGS+=("--max-samples" "$2")
      shift 2
      ;;
    --offset)
      OFFSET="$2"
      FORWARD_ARGS+=("--offset" "$2")
      shift 2
      ;;
    --max-prompt-length)
      MAX_PROMPT_LENGTH="$2"
      FORWARD_ARGS+=("--max-prompt-length" "$2")
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      FORWARD_ARGS+=("--max-new-tokens" "$2")
      shift 2
      ;;
    --repetition-penalty)
      REPETITION_PENALTY="$2"
      FORWARD_ARGS+=("--repetition-penalty" "$2")
      shift 2
      ;;
    --sid-levels)
      SID_LEVELS="$2"
      FORWARD_ARGS+=("--sid-levels" "$2")
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      FORWARD_ARGS+=("--cache-dir" "$2")
      shift 2
      ;;
    --reuse-summary-path)
      REUSE_SUMMARY_PATH="$2"
      FORWARD_ARGS+=("--reuse-summary-path" "$2")
      shift 2
      ;;
    --reuse-details-path)
      REUSE_DETAILS_PATH="$2"
      FORWARD_ARGS+=("--reuse-details-path" "$2")
      shift 2
      ;;
    --disable-cache-reuse)
      DISABLE_CACHE_REUSE=1
      FORWARD_ARGS+=("--disable-cache-reuse")
      shift
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1
      FORWARD_ARGS+=("--trust-remote-code")
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      FORWARD_ARGS+=("--python-bin" "$2")
      shift 2
      ;;
    --conda-activate)
      CONDA_ACTIVATE="$2"
      FORWARD_ARGS+=("--conda-activate" "$2")
      shift 2
      ;;
    --conda-env)
      CONDA_ENV_NAME="$2"
      FORWARD_ARGS+=("--conda-env" "$2")
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

if [[ -z "$SUMMARY_PATH" ]]; then
  SUMMARY_PATH="${SUMMARY_DIR_DEFAULT}/$(sanitize_name "${RUN_NAME}")_${TS}_summary.json"
fi
if [[ -z "$DETAILS_PATH" ]]; then
  DETAILS_PATH="${SUMMARY_DIR_DEFAULT}/$(sanitize_name "${RUN_NAME}")_${TS}_details.json"
fi

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
    --summary-path "$SUMMARY_PATH"
    --details-path "$DETAILS_PATH"
    --python-bin "$PYTHON_BIN"
    --conda-activate "$CONDA_ACTIVATE"
    --conda-env "$CONDA_ENV_NAME"
  )
  if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    CHILD_ARGS+=("${FORWARD_ARGS[@]}")
  fi
  nohup bash "$SCRIPT_PATH" "${CHILD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[INFO] Analysis started in background. pid=$PID"
  echo "[INFO] Log file: $LOG_FILE"
  if [[ "$MODE" == "nohup" ]]; then
    echo "[INFO] Press Ctrl-C to stop following logs (analysis keeps running)."
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

require_exists "$REPO_ROOT" "REPO_ROOT"
cd "$REPO_ROOT"

require_file "${REPO_ROOT}/analyze_rl_beam_hint.py" "analyze_rl_beam_hint.py"
require_exists "$MODEL_PATH" "model path"
require_file "${DATA_DIR}/train.json" "RL train dataset"
require_file "$INDEX_PATH" "id2sid index file"
if [[ -n "$ADD_TOKENS_PATH" ]]; then
  require_file "$ADD_TOKENS_PATH" "new_tokens.json"
fi

ANALYZE_CMD=(
  "$PYTHON_BIN"
  analyze_rl_beam_hint.py
  --model-path "$MODEL_PATH"
  --data-dir "$DATA_DIR"
  --index-path "$INDEX_PATH"
  --add-tokens-path "$ADD_TOKENS_PATH"
  --summary-path "$SUMMARY_PATH"
  --details-path "$DETAILS_PATH"
  --beam-sizes "$BEAM_SIZES"
  --hint-depth "$HINT_DEPTH"
  --max-hint-depth "$MAX_HINT_DEPTH"
  --batch-size "$BATCH_SIZE"
  --offset "$OFFSET"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --repetition-penalty "$REPETITION_PENALTY"
  --sid-levels "$SID_LEVELS"
  --cache-dir "$CACHE_DIR"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  ANALYZE_CMD+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "$REUSE_SUMMARY_PATH" ]]; then
  ANALYZE_CMD+=(--reuse-summary-path "$REUSE_SUMMARY_PATH")
fi
if [[ -n "$REUSE_DETAILS_PATH" ]]; then
  ANALYZE_CMD+=(--reuse-details-path "$REUSE_DETAILS_PATH")
fi
if [[ "$DISABLE_CACHE_REUSE" -eq 1 ]]; then
  ANALYZE_CMD+=(--disable-cache-reuse)
fi
if [[ "$TRUST_REMOTE_CODE" -eq 1 ]]; then
  ANALYZE_CMD+=(--trust-remote-code)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${ANALYZE_CMD[@]}"
  echo
  exit 0
fi

mkdir -p "$(dirname -- "$LOG_FILE")"
mkdir -p "$(dirname -- "$SUMMARY_PATH")"
mkdir -p "$(dirname -- "$DETAILS_PATH")"
if [[ "$FROM_NOHUP" -eq 0 ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] INDEX_PATH=$INDEX_PATH"
echo "[INFO] ADD_TOKENS_PATH=$ADD_TOKENS_PATH"
echo "[INFO] SUMMARY_PATH=$SUMMARY_PATH"
echo "[INFO] DETAILS_PATH=$DETAILS_PATH"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] RUN_NAME=$RUN_NAME"
echo "[INFO] BEAM_SIZES=$BEAM_SIZES"
echo "[INFO] HINT_DEPTH=$HINT_DEPTH"
echo "[INFO] MAX_HINT_DEPTH=$MAX_HINT_DEPTH"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE"
echo "[INFO] MAX_SAMPLES=${MAX_SAMPLES:-<all>}"
echo "[INFO] OFFSET=$OFFSET"
echo "[INFO] MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH"
echo "[INFO] MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "[INFO] REPETITION_PENALTY=$REPETITION_PENALTY"
echo "[INFO] SID_LEVELS=$SID_LEVELS"
echo "[INFO] CACHE_DIR=$CACHE_DIR"
echo "[INFO] REUSE_SUMMARY_PATH=${REUSE_SUMMARY_PATH:-<auto>}"
echo "[INFO] REUSE_DETAILS_PATH=${REUSE_DETAILS_PATH:-<auto>}"
echo "[INFO] DISABLE_CACHE_REUSE=$DISABLE_CACHE_REUSE"
echo "[INFO] TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE"

"${ANALYZE_CMD[@]}"
