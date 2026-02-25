#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/evaluate_sft_3b.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_LIST="${CUDA_LIST:-0 1 2 3}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
AUTO_DATA_MAPPING="${AUTO_DATA_MAPPING:-1}"

# run mode:
# - background (default): spawn nohup background worker and tail log in current terminal
# - foreground: run in current shell
RUN_MODE="${RUN_MODE:-background}"
TAIL_LOG="${TAIL_LOG:-1}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/evaluate_all_checkpoints_$(date +%Y%m%d_%H%M%S).log}"
_BG_CHILD="${_BG_CHILD:-0}"

SFT_ROOT="${SFT_ROOT:-$REPO_ROOT/saves/qwen2.5-3b/full}"
RL_ROOT="${RL_ROOT:-$REPO_ROOT/rl_outputs}"
INCLUDE_SFT="${INCLUDE_SFT:-1}"
INCLUDE_RL="${INCLUDE_RL:-1}"

# optional comma-separated substring filter, e.g.
# MODEL_FILTER="Industrial_and_Scientific,Instruments-grec"
MODEL_FILTER="${MODEL_FILTER:-}"

# 1: evaluate all checkpoints even if metrics.json already exists
FORCE_REEVAL="${FORCE_REEVAL:-0}"
# 1: print plan only
DRY_RUN="${DRY_RUN:-0}"

# Default data mapping.
# You can override these via env on your machine.
INDUSTRIAL_TEST_DATA_PATH="${INDUSTRIAL_TEST_DATA_PATH:-$DATA_ROOT/Industrial_and_Scientific/sft/test.json}"
INDUSTRIAL_INDEX_PATH="${INDUSTRIAL_INDEX_PATH:-$DATA_ROOT/Industrial_and_Scientific/id2sid.json}"

INSTRUMENTS_TEST_DATA_PATH="${INSTRUMENTS_TEST_DATA_PATH:-$DATA_ROOT/Instruments/sft/test.json}"
INSTRUMENTS_INDEX_PATH="${INSTRUMENTS_INDEX_PATH:-$DATA_ROOT/Instruments/id2sid.json}"

INSTRUMENTS_GREC_TEST_DATA_PATH="${INSTRUMENTS_GREC_TEST_DATA_PATH:-$INSTRUMENTS_TEST_DATA_PATH}"
INSTRUMENTS_GREC_INDEX_PATH="${INSTRUMENTS_GREC_INDEX_PATH:-$INSTRUMENTS_INDEX_PATH}"

INSTRUMENTS_MIMIONEREC_TEST_DATA_PATH="${INSTRUMENTS_MIMIONEREC_TEST_DATA_PATH:-$INSTRUMENTS_TEST_DATA_PATH}"
INSTRUMENTS_MIMIONEREC_INDEX_PATH="${INSTRUMENTS_MIMIONEREC_INDEX_PATH:-$INSTRUMENTS_INDEX_PATH}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate_all_checkpoints.sh

Environment overrides:
  EVAL_SCRIPT=./evaluate_sft_3b.sh
  PYTHON_BIN=python
  CUDA_LIST="0 1 2 3"
  DATA_ROOT=./data
  AUTO_DATA_MAPPING=1
  RUN_MODE=background   # background | foreground
  TAIL_LOG=1            # background mode only, 1 means tail -f after launch
  LOG_DIR=./log
  LOG_FILE=./log/evaluate_all_checkpoints_xxx.log
  SFT_ROOT=./saves/qwen2.5-3b/full
  RL_ROOT=./rl_outputs
  INCLUDE_SFT=1
  INCLUDE_RL=1
  MODEL_FILTER="Industrial_and_Scientific,Instruments-grec"
  FORCE_REEVAL=0
  DRY_RUN=0

  # Data mapping (auto-selected by model name prefix)
  INDUSTRIAL_TEST_DATA_PATH=...
  INDUSTRIAL_INDEX_PATH=...
  INSTRUMENTS_TEST_DATA_PATH=...
  INSTRUMENTS_INDEX_PATH=...
  INSTRUMENTS_GREC_TEST_DATA_PATH=...
  INSTRUMENTS_GREC_INDEX_PATH=...
  INSTRUMENTS_MIMIONEREC_TEST_DATA_PATH=...
  INSTRUMENTS_MIMIONEREC_INDEX_PATH=...

Examples:
  # 先看计划，不实际跑
  DRY_RUN=1 bash scripts/evaluate_all_checkpoints.sh

  # 只使用自动数据匹配（按模型名 cb 宽度匹配 data variant）
  AUTO_DATA_MAPPING=1 DRY_RUN=1 bash scripts/evaluate_all_checkpoints.sh

  # 只评测 grecckpt
  MODEL_FILTER="Instruments-grec" bash scripts/evaluate_all_checkpoints.sh

  # 强制重跑全部 checkpoint
  FORCE_REEVAL=1 bash scripts/evaluate_all_checkpoints.sh

  # 前台运行（不后台）
  RUN_MODE=foreground bash scripts/evaluate_all_checkpoints.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$RUN_MODE" == "background" && "$_BG_CHILD" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  touch "$LOG_FILE"

  echo "[INFO] launch_mode=background"
  echo "[INFO] log_file=$LOG_FILE"
  nohup env _BG_CHILD=1 RUN_MODE=foreground LOG_FILE="$LOG_FILE" \
    bash "$0" "$@" >> "$LOG_FILE" 2>&1 &
  bg_pid=$!
  echo "[INFO] background_pid=$bg_pid"
  echo "$bg_pid" > "${LOG_FILE}.pid"
  echo "[INFO] pid_file=${LOG_FILE}.pid"

  if [[ "$TAIL_LOG" == "1" ]]; then
    echo "[INFO] tailing log (Ctrl+C to detach): $LOG_FILE"
    tail -f "$LOG_FILE"
  else
    echo "[INFO] TAIL_LOG=0, not tailing."
  fi
  exit 0
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] EVAL_SCRIPT not found: $EVAL_SCRIPT"
  exit 1
fi

has_checkpoints() {
  local model_root="$1"
  find "$model_root" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .
}

extract_cb_width() {
  local model_name="$1"
  if [[ "$model_name" =~ cb([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$model_name" =~ -4-([0-9]+)(-|$) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

dir_mtime_epoch() {
  local path="$1"
  if stat -c %Y "$path" >/dev/null 2>&1; then
    stat -c %Y "$path"
  else
    stat -f %m "$path"
  fi
}

pick_latest_variant_dir_by_cb() {
  local variant_prefix="$1"
  local cb_width="$2"
  local best_dir=""
  local best_mtime=0
  local candidate candidate_name candidate_mtime

  [[ -d "$DATA_ROOT" ]] || return 1

  while IFS= read -r candidate; do
    candidate_name="$(basename "$candidate")"
    [[ "$candidate_name" == *"_cb${cb_width}-"* ]] || continue

    candidate_mtime="$(dir_mtime_epoch "$candidate")"
    if [[ -z "$best_dir" || "$candidate_mtime" -gt "$best_mtime" ]]; then
      best_dir="$candidate"
      best_mtime="$candidate_mtime"
    fi
  done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${variant_prefix}_index_emb-*")

  if [[ -n "$best_dir" ]]; then
    echo "$best_dir"
    return 0
  fi
  return 1
}

matches_filter() {
  local model_name="$1"
  if [[ -z "$MODEL_FILTER" ]]; then
    return 0
  fi
  local token
  IFS=',' read -r -a _filters <<< "$MODEL_FILTER"
  for token in "${_filters[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -z "$token" ]] && continue
    if [[ "$model_name" == *"$token"* ]]; then
      return 0
    fi
  done
  return 1
}

resolve_eval_profile() {
  local model_name="$1"
  RES_CATEGORY=""
  RES_TEST_DATA_PATH=""
  RES_INDEX_PATH=""
  RES_PROFILE_INFO=""
  RES_CB_WIDTH=""

  local cb_width=""
  cb_width="$(extract_cb_width "$model_name" || true)"
  RES_CB_WIDTH="${cb_width:-n/a}"

  if [[ "$model_name" == Industrial_and_Scientific* ]]; then
    RES_CATEGORY="Industrial_and_Scientific"
    RES_TEST_DATA_PATH="$INDUSTRIAL_TEST_DATA_PATH"
    RES_INDEX_PATH="$INDUSTRIAL_INDEX_PATH"
    RES_PROFILE_INFO="fixed:industrial_default"
    return
  fi

  if [[ "$model_name" == Instruments-grec* || "$model_name" == Instruments_grec* ]]; then
    RES_CATEGORY="Instruments_grec"

    if [[ "$AUTO_DATA_MAPPING" == "1" && -n "$cb_width" ]]; then
      local variant_dir
      variant_dir="$(pick_latest_variant_dir_by_cb "Instruments_grec" "$cb_width" || true)"
      if [[ -n "$variant_dir" ]]; then
        local candidate_test candidate_index
        candidate_test="$variant_dir/sft/test.json"
        candidate_index="$variant_dir/id2sid.json"
        if [[ -f "$candidate_test" && -f "$candidate_index" ]]; then
          RES_TEST_DATA_PATH="$candidate_test"
          RES_INDEX_PATH="$candidate_index"
          RES_PROFILE_INFO="auto:variant_dir=$variant_dir"
          return
        fi
      fi
    fi

    RES_TEST_DATA_PATH="$INSTRUMENTS_GREC_TEST_DATA_PATH"
    RES_INDEX_PATH="$INSTRUMENTS_GREC_INDEX_PATH"
    RES_PROFILE_INFO="fallback:env_INSTRUMENTS_GREC_*"
    return
  fi

  if [[ "$model_name" == Instruments-mimionerec* || "$model_name" == Instruments_mimionerec* ]]; then
    RES_CATEGORY="Instruments_mimionerec"

    if [[ "$AUTO_DATA_MAPPING" == "1" && -n "$cb_width" ]]; then
      local variant_dir
      variant_dir="$(pick_latest_variant_dir_by_cb "Instruments_mimionerec" "$cb_width" || true)"
      if [[ -n "$variant_dir" ]]; then
        local candidate_test candidate_index
        candidate_test="$variant_dir/sft/test.json"
        candidate_index="$variant_dir/id2sid.json"
        if [[ -f "$candidate_test" && -f "$candidate_index" ]]; then
          RES_TEST_DATA_PATH="$candidate_test"
          RES_INDEX_PATH="$candidate_index"
          RES_PROFILE_INFO="auto:variant_dir=$variant_dir"
          return
        fi
      fi
    fi

    RES_TEST_DATA_PATH="$INSTRUMENTS_MIMIONEREC_TEST_DATA_PATH"
    RES_INDEX_PATH="$INSTRUMENTS_MIMIONEREC_INDEX_PATH"
    RES_PROFILE_INFO="fallback:env_INSTRUMENTS_MIMIONEREC_*"
    return
  fi

  if [[ "$model_name" == Instruments* ]]; then
    RES_CATEGORY="Instruments"
    RES_TEST_DATA_PATH="$INSTRUMENTS_TEST_DATA_PATH"
    RES_INDEX_PATH="$INSTRUMENTS_INDEX_PATH"
    RES_PROFILE_INFO="fixed:instruments_default"
    return
  fi

  # fallback
  RES_CATEGORY="Industrial_and_Scientific"
  RES_TEST_DATA_PATH="$INDUSTRIAL_TEST_DATA_PATH"
  RES_INDEX_PATH="$INDUSTRIAL_INDEX_PATH"
  RES_PROFILE_INFO="fallback:industrial_default"
}

collect_model_roots() {
  local base_dir="$1"
  if [[ ! -d "$base_dir" ]]; then
    return
  fi
  find "$base_dir" -mindepth 1 -maxdepth 1 -type d | sort -V
}

collect_pending_ckpts() {
  local model_root="$1"
  local model_name="$2"
  PENDING_CKPTS=()

  local ckpt_path ckpt_name metrics_path
  while IFS= read -r ckpt_path; do
    ckpt_name="$(basename "$ckpt_path")"
    if [[ "$FORCE_REEVAL" == "1" ]]; then
      PENDING_CKPTS+=("$ckpt_name")
      continue
    fi
    metrics_path="$REPO_ROOT/results/$model_name/$ckpt_name/metrics.json"
    if [[ ! -f "$metrics_path" ]]; then
      PENDING_CKPTS+=("$ckpt_name")
    fi
  done < <(find "$model_root" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
}

MODEL_ROOTS=()

if [[ "$INCLUDE_SFT" == "1" ]]; then
  while IFS= read -r path; do
    MODEL_ROOTS+=("$path")
  done < <(collect_model_roots "$SFT_ROOT")
fi

if [[ "$INCLUDE_RL" == "1" ]]; then
  while IFS= read -r path; do
    MODEL_ROOTS+=("$path")
  done < <(collect_model_roots "$RL_ROOT")
fi

if [[ ${#MODEL_ROOTS[@]} -eq 0 ]]; then
  echo "[ERROR] no candidate model roots found."
  echo "        SFT_ROOT=$SFT_ROOT"
  echo "        RL_ROOT=$RL_ROOT"
  exit 1
fi

echo "[INFO] repo_root=$REPO_ROOT"
echo "[INFO] eval_script=$EVAL_SCRIPT"
echo "[INFO] include_sft=$INCLUDE_SFT sft_root=$SFT_ROOT"
echo "[INFO] include_rl=$INCLUDE_RL rl_root=$RL_ROOT"
echo "[INFO] data_root=$DATA_ROOT auto_data_mapping=$AUTO_DATA_MAPPING"
echo "[INFO] model_filter=${MODEL_FILTER:-<empty>}"
echo "[INFO] force_reeval=$FORCE_REEVAL dry_run=$DRY_RUN"
echo

total_models=0
scheduled_models=0
scheduled_ckpts=0

for model_root in "${MODEL_ROOTS[@]}"; do
  model_name="$(basename "$model_root")"
  ((total_models += 1))

  if ! matches_filter "$model_name"; then
    continue
  fi

  if ! has_checkpoints "$model_root"; then
    continue
  fi

  collect_pending_ckpts "$model_root" "$model_name"
  if [[ ${#PENDING_CKPTS[@]} -eq 0 ]]; then
    echo "[SKIP] $model_name (all checkpoints already have metrics.json)"
    continue
  fi

  resolve_eval_profile "$model_name"

  if [[ ! -f "$RES_TEST_DATA_PATH" ]]; then
    echo "[ERROR] test_data_path not found for model=$model_name: $RES_TEST_DATA_PATH"
    exit 1
  fi
  if [[ ! -f "$RES_INDEX_PATH" ]]; then
    echo "[ERROR] index_path not found for model=$model_name: $RES_INDEX_PATH"
    exit 1
  fi

  ckpt_list_str="$(printf '%s ' "${PENDING_CKPTS[@]}")"
  ckpt_list_str="${ckpt_list_str% }"

  echo "[PLAN] model=$model_name"
  echo "       root=$model_root"
  echo "       category=$RES_CATEGORY"
  echo "       data_profile=$RES_PROFILE_INFO"
  echo "       cb_width=$RES_CB_WIDTH"
  echo "       test_data=$RES_TEST_DATA_PATH"
  echo "       index=$RES_INDEX_PATH"
  echo "       pending_ckpts=$ckpt_list_str"

  ((scheduled_models += 1))
  ((scheduled_ckpts+=${#PENDING_CKPTS[@]}))

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  CATEGORY="$RES_CATEGORY" \
  TEST_DATA_PATH="$RES_TEST_DATA_PATH" \
  INDEX_PATH="$RES_INDEX_PATH" \
  CUDA_LIST="$CUDA_LIST" \
  PYTHON_BIN="$PYTHON_BIN" \
  CKPT_LIST="$ckpt_list_str" \
  bash "$EVAL_SCRIPT" "$model_root"

  echo "[DONE] model=$model_name"
  echo
done

echo "[SUMMARY] total_models_seen=$total_models scheduled_models=$scheduled_models scheduled_ckpts=$scheduled_ckpts"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[SUMMARY] dry-run only, no evaluation executed."
fi
