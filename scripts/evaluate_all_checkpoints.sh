#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SIDECAR_PY="${SIDECAR_PY:-$REPO_ROOT/scripts/evaluate_all_checkpoints_sidecar.py}"
MANAGER_DIR="${MANAGER_DIR:-$REPO_ROOT/log/evaluate_all_checkpoints}"

usage() {
  cat <<'EOF'
Usage:
  # Legacy one-shot scan/eval (existing behavior)
  bash scripts/evaluate_all_checkpoints.sh
  bash scripts/evaluate_all_checkpoints.sh once

  # Long-running watcher
  bash scripts/evaluate_all_checkpoints.sh run [--instance NAME] [sidecar args...]
  bash scripts/evaluate_all_checkpoints.sh start [--instance NAME] [sidecar args...]
  bash scripts/evaluate_all_checkpoints.sh stop [--instance NAME]
  bash scripts/evaluate_all_checkpoints.sh status [--instance NAME]
  bash scripts/evaluate_all_checkpoints.sh tail [--instance NAME]

Watcher examples:
  # Foreground watcher
  bash scripts/evaluate_all_checkpoints.sh run

  # Background watcher
  bash scripts/evaluate_all_checkpoints.sh start --instance remote_eval

  # Tighten readiness gate and shorten poll interval
  bash scripts/evaluate_all_checkpoints.sh run \
    --poll-interval-seconds 30 \
    --stable-age-seconds 300 \
    --stable-confirmation-polls 2

  # Keep the watcher attached to GPUs while idle
  IDLE_HOLD_MEMORY_RATIO=0.95 \
  bash scripts/evaluate_all_checkpoints.sh run

Legacy environment overrides:
  EVAL_SCRIPT=./evaluate_sft_3b.sh
  PYTHON_BIN=python
  CUDA_LIST="0 1 2 3"
  DATA_ROOT=./data
  AUTO_DATA_MAPPING=1
  RUN_MODE=background   # legacy one-shot only: background | foreground
  TAIL_LOG=1            # legacy one-shot only, background mode
  LOG_DIR=./log
  LOG_FILE=./log/evaluate_all_checkpoints_xxx.log
  SFT_ROOT=./saves/qwen2.5-3b/full
  RL_ROOT=./rl_outputs
  INCLUDE_SFT=1
  INCLUDE_RL=1
  MODEL_FILTER="Industrial_and_Scientific,Instruments-grec"
  FORCE_REEVAL=0
  DRY_RUN=0

Watcher environment overrides:
  SIDECAR_PY=./scripts/evaluate_all_checkpoints_sidecar.py
  MANAGER_DIR=./log/evaluate_all_checkpoints
  POLL_INTERVAL_SECONDS=60
  STABLE_AGE_SECONDS=180
  STABLE_CONFIRMATION_POLLS=2
  IDLE_HOLD_ENABLED=1
  IDLE_HOLD_MEMORY_RATIO=0.95
  IDLE_HOLD_RELEASE_GRACE_SECONDS=5
  WATCH_STATE_PATH=state/evaluate_all_checkpoints/watch_state.json
EOF
}

require_sidecar() {
  if [[ ! -f "$SIDECAR_PY" ]]; then
    echo "[ERROR] sidecar script not found: $SIDECAR_PY"
    exit 1
  fi
}

is_running() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  kill -0 "$pid" 2>/dev/null
}

run_legacy_once() {
  local eval_script="${EVAL_SCRIPT:-$REPO_ROOT/evaluate_sft_3b.sh}"
  local python_bin="${PYTHON_BIN:-python}"
  local cuda_list="${CUDA_LIST:-0 1 2 3}"
  local data_root="${DATA_ROOT:-$REPO_ROOT/data}"
  local auto_data_mapping="${AUTO_DATA_MAPPING:-1}"

  local run_mode="${RUN_MODE:-background}"
  local tail_log="${TAIL_LOG:-1}"
  local log_dir="${LOG_DIR:-$REPO_ROOT/log}"
  local log_file="${LOG_FILE:-$log_dir/evaluate_all_checkpoints_$(date +%Y%m%d_%H%M%S).log}"
  local bg_child="${_BG_CHILD:-0}"

  local sft_root="${SFT_ROOT:-$REPO_ROOT/saves/qwen2.5-3b/full}"
  local rl_root="${RL_ROOT:-$REPO_ROOT/rl_outputs}"
  local include_sft="${INCLUDE_SFT:-1}"
  local include_rl="${INCLUDE_RL:-1}"
  local model_filter="${MODEL_FILTER:-}"
  local force_reeval="${FORCE_REEVAL:-0}"
  local dry_run="${DRY_RUN:-0}"

  local industrial_test_data_path="${INDUSTRIAL_TEST_DATA_PATH:-$data_root/Industrial_and_Scientific/sft/test.json}"
  local industrial_index_path="${INDUSTRIAL_INDEX_PATH:-$data_root/Industrial_and_Scientific/id2sid.json}"

  local instruments_test_data_path="${INSTRUMENTS_TEST_DATA_PATH:-$data_root/Instruments/sft/test.json}"
  local instruments_index_path="${INSTRUMENTS_INDEX_PATH:-$data_root/Instruments/id2sid.json}"

  local games_test_data_path="${GAMES_TEST_DATA_PATH:-$data_root/Games/sft/test.json}"
  local games_index_path="${GAMES_INDEX_PATH:-$data_root/Games/id2sid.json}"

  local arts_test_data_path="${ARTS_TEST_DATA_PATH:-$data_root/Arts/sft/test.json}"
  local arts_index_path="${ARTS_INDEX_PATH:-$data_root/Arts/id2sid.json}"

  local instruments_grec_fallback_variant_dir="${INSTRUMENTS_GREC_FALLBACK_VARIANT_DIR:-$data_root/Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47}"
  local instruments_grec_compact_variant_dir="${INSTRUMENTS_GREC_COMPACT_VARIANT_DIR:-$data_root/Instruments_grec_index}"
  local instruments_grec_test_data_path="${INSTRUMENTS_GREC_TEST_DATA_PATH:-$instruments_grec_fallback_variant_dir/sft/test.json}"
  local instruments_grec_index_path="${INSTRUMENTS_GREC_INDEX_PATH:-$instruments_grec_fallback_variant_dir/id2sid.json}"

  local instruments_mimionerec_test_data_path="${INSTRUMENTS_MIMIONEREC_TEST_DATA_PATH:-$instruments_test_data_path}"
  local instruments_mimionerec_index_path="${INSTRUMENTS_MIMIONEREC_INDEX_PATH:-$instruments_index_path}"

  if [[ "$run_mode" == "background" && "$bg_child" != "1" ]]; then
    mkdir -p "$log_dir"
    touch "$log_file"

    echo "[INFO] launch_mode=background"
    echo "[INFO] log_file=$log_file"
    nohup env _BG_CHILD=1 RUN_MODE=foreground LOG_FILE="$log_file" \
      bash "$0" once "$@" >> "$log_file" 2>&1 &
    local legacy_pid=$!
    echo "[INFO] background_pid=$legacy_pid"
    echo "$legacy_pid" > "${log_file}.pid"
    echo "[INFO] pid_file=${log_file}.pid"

    if [[ "$tail_log" == "1" ]]; then
      echo "[INFO] tailing log (Ctrl+C to detach): $log_file"
      tail -f "$log_file"
    else
      echo "[INFO] TAIL_LOG=0, not tailing."
    fi
    return 0
  fi

  if [[ ! -f "$eval_script" ]]; then
    echo "[ERROR] EVAL_SCRIPT not found: $eval_script"
    return 1
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

  candidate_variant_dirs_for_prefix() {
    local variant_prefix="$1"
    [[ -d "$data_root" ]] || return 0
    find "$data_root" -mindepth 1 -maxdepth 1 -type d \( \
      -name "${variant_prefix}_index_emb-*" -o \
      -name "${variant_prefix}_index" -o \
      -name "${variant_prefix}_*" \
    \) | sort -V
  }

  pick_latest_variant_dir_by_cb() {
    local variant_prefix="$1"
    local cb_width="$2"
    local best_dir=""
    local best_mtime=0
    local candidate candidate_name candidate_mtime

    [[ -d "$data_root" ]] || return 1

    while IFS= read -r candidate; do
      candidate_name="$(basename "$candidate")"
      if [[ "$candidate_name" == *"_cb${cb_width}-"* || "$candidate_name" == *"_cb${cb_width}_"* || "$candidate_name" == *"_cb${cb_width}" ]]; then
        :
      elif [[ "$candidate_name" == "${variant_prefix}_index" && "$cb_width" == "256" ]]; then
        :
      else
        continue
      fi

      candidate_mtime="$(dir_mtime_epoch "$candidate")"
      if [[ -z "$best_dir" || "$candidate_mtime" -gt "$best_mtime" ]]; then
        best_dir="$candidate"
        best_mtime="$candidate_mtime"
      fi
    done < <(candidate_variant_dirs_for_prefix "$variant_prefix")

    if [[ -n "$best_dir" ]]; then
      echo "$best_dir"
      return 0
    fi
    return 1
  }

  pick_latest_variant_dir_by_prefix() {
    local variant_prefix="$1"
    local best_dir=""
    local best_mtime=0
    local candidate candidate_mtime

    [[ -d "$data_root" ]] || return 1

    while IFS= read -r candidate; do
      candidate_mtime="$(dir_mtime_epoch "$candidate")"
      if [[ -z "$best_dir" || "$candidate_mtime" -gt "$best_mtime" ]]; then
        best_dir="$candidate"
        best_mtime="$candidate_mtime"
      fi
    done < <(candidate_variant_dirs_for_prefix "$variant_prefix")

    if [[ -n "$best_dir" ]]; then
      echo "$best_dir"
      return 0
    fi
    return 1
  }

  resolve_base_category_from_model_name() {
    local model_name="$1"
    local normalized="${model_name,,}"
    case "$normalized" in
      industrial_and_scientific*|industrial-and-scientific*|industrial*|ind|ind-*|ind_*|ias|ias-*|ias_*)
        echo "Industrial_and_Scientific"
        return 0
        ;;
      instruments*|instrument*|ins|ins-*|ins_*)
        echo "Instruments"
        return 0
        ;;
      games*|game*|gam|gam-*|gam_*|gms|gms-*|gms_*)
        echo "Games"
        return 0
        ;;
      arts*|art|art-*|art_*)
        echo "Arts"
        return 0
        ;;
    esac
    return 1
  }

  matches_filter() {
    local model_name="$1"
    if [[ -z "$model_filter" ]]; then
      return 0
    fi
    local token
    IFS=',' read -r -a _filters <<< "$model_filter"
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

    local base_category=""
    base_category="$(resolve_base_category_from_model_name "$model_name" || true)"
    local normalized_model_name="${model_name,,}"

    if [[ "$base_category" == "Industrial_and_Scientific" ]]; then
      RES_CATEGORY="Industrial_and_Scientific"
      RES_TEST_DATA_PATH="$industrial_test_data_path"
      RES_INDEX_PATH="$industrial_index_path"
      RES_PROFILE_INFO="fixed:industrial_default"
      return
    fi

    if [[ "$normalized_model_name" == ins-lc* ]]; then
      RES_CATEGORY="Instruments_grec"
      if [[ -f "$instruments_grec_compact_variant_dir/sft/test.json" && -f "$instruments_grec_compact_variant_dir/id2sid.json" ]]; then
        RES_TEST_DATA_PATH="$instruments_grec_compact_variant_dir/sft/test.json"
        RES_INDEX_PATH="$instruments_grec_compact_variant_dir/id2sid.json"
        RES_PROFILE_INFO="fixed:legacy_ins_lc_compact_variant_dir=Instruments_grec_index"
      else
        RES_TEST_DATA_PATH="$instruments_grec_test_data_path"
        RES_INDEX_PATH="$instruments_grec_index_path"
        RES_PROFILE_INFO="fallback:legacy_ins_lc_fixed_grec_cb256"
      fi
      return
    fi

    if [[ -n "$base_category" && ( "$model_name" == "${base_category}-grec"* || "$model_name" == "${base_category}_grec"* ) ]]; then
      RES_CATEGORY="${base_category}_grec"

      if [[ "$auto_data_mapping" == "1" && -n "$cb_width" ]]; then
        local variant_dir
        variant_dir="$(pick_latest_variant_dir_by_cb "${RES_CATEGORY}" "$cb_width" || true)"
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

      if [[ "$base_category" == "Instruments" && -z "$cb_width" ]]; then
        RES_TEST_DATA_PATH="$instruments_grec_test_data_path"
        RES_INDEX_PATH="$instruments_grec_index_path"
        if [[ "$RES_TEST_DATA_PATH" == "$instruments_grec_fallback_variant_dir/sft/test.json" && "$RES_INDEX_PATH" == "$instruments_grec_fallback_variant_dir/id2sid.json" ]]; then
          RES_PROFILE_INFO="fallback:fixed_grec_cb256"
        else
          RES_PROFILE_INFO="fallback:env_INSTRUMENTS_GREC_*"
        fi
        return
      fi

      local latest_variant_dir=""
      latest_variant_dir="$(pick_latest_variant_dir_by_prefix "${RES_CATEGORY}" || true)"
      if [[ -n "$latest_variant_dir" ]]; then
        local candidate_test candidate_index
        candidate_test="$latest_variant_dir/sft/test.json"
        candidate_index="$latest_variant_dir/id2sid.json"
        if [[ -f "$candidate_test" && -f "$candidate_index" ]]; then
          RES_TEST_DATA_PATH="$candidate_test"
          RES_INDEX_PATH="$candidate_index"
          RES_PROFILE_INFO="fallback:latest_variant_dir=$latest_variant_dir"
          return
        fi
      fi

      if [[ "$base_category" == "Instruments" ]]; then
        RES_TEST_DATA_PATH="$instruments_grec_test_data_path"
        RES_INDEX_PATH="$instruments_grec_index_path"
        if [[ "$RES_TEST_DATA_PATH" == "$instruments_grec_fallback_variant_dir/sft/test.json" && "$RES_INDEX_PATH" == "$instruments_grec_fallback_variant_dir/id2sid.json" ]]; then
          RES_PROFILE_INFO="fallback:fixed_grec_cb256"
        else
          RES_PROFILE_INFO="fallback:env_INSTRUMENTS_GREC_*"
        fi
        return
      fi
      return
    fi

    if [[ -n "$base_category" && ( "$model_name" == "${base_category}-mimionerec"* || "$model_name" == "${base_category}_mimionerec"* ) ]]; then
      RES_CATEGORY="${base_category}_mimionerec"

      if [[ "$auto_data_mapping" == "1" && -n "$cb_width" ]]; then
        local variant_dir
        variant_dir="$(pick_latest_variant_dir_by_cb "${RES_CATEGORY}" "$cb_width" || true)"
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

      local latest_variant_dir=""
      latest_variant_dir="$(pick_latest_variant_dir_by_prefix "${RES_CATEGORY}" || true)"
      if [[ -n "$latest_variant_dir" ]]; then
        local candidate_test candidate_index
        candidate_test="$latest_variant_dir/sft/test.json"
        candidate_index="$latest_variant_dir/id2sid.json"
        if [[ -f "$candidate_test" && -f "$candidate_index" ]]; then
          RES_TEST_DATA_PATH="$candidate_test"
          RES_INDEX_PATH="$candidate_index"
          RES_PROFILE_INFO="fallback:latest_variant_dir=$latest_variant_dir"
          return
        fi
      fi

      if [[ "$base_category" == "Instruments" ]]; then
        RES_TEST_DATA_PATH="$instruments_mimionerec_test_data_path"
        RES_INDEX_PATH="$instruments_mimionerec_index_path"
        RES_PROFILE_INFO="fallback:env_INSTRUMENTS_MIMIONEREC_*"
        return
      fi
      return
    fi

    if [[ "$base_category" == "Instruments" ]]; then
      RES_CATEGORY="Instruments"
      RES_TEST_DATA_PATH="$instruments_test_data_path"
      RES_INDEX_PATH="$instruments_index_path"
      RES_PROFILE_INFO="fixed:instruments_default"
      return
    fi

    if [[ "$base_category" == "Games" ]]; then
      RES_CATEGORY="Games"
      RES_TEST_DATA_PATH="$games_test_data_path"
      RES_INDEX_PATH="$games_index_path"
      RES_PROFILE_INFO="fixed:games_default"
      return
    fi

    if [[ "$base_category" == "Arts" ]]; then
      RES_CATEGORY="Arts"
      RES_TEST_DATA_PATH="$arts_test_data_path"
      RES_INDEX_PATH="$arts_index_path"
      RES_PROFILE_INFO="fixed:arts_default"
      return
    fi

    RES_CATEGORY="Industrial_and_Scientific"
    RES_TEST_DATA_PATH="$industrial_test_data_path"
    RES_INDEX_PATH="$industrial_index_path"
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
      if [[ "$force_reeval" == "1" ]]; then
        PENDING_CKPTS+=("$ckpt_name")
        continue
      fi
      metrics_path="$REPO_ROOT/results/$model_name/$ckpt_name/metrics.json"
      if [[ ! -f "$metrics_path" ]]; then
        PENDING_CKPTS+=("$ckpt_name")
      fi
    done < <(find "$model_root" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  }

  local model_roots=()
  if [[ "$include_sft" == "1" ]]; then
    while IFS= read -r path; do
      model_roots+=("$path")
    done < <(collect_model_roots "$sft_root")
  fi

  if [[ "$include_rl" == "1" ]]; then
    while IFS= read -r path; do
      model_roots+=("$path")
    done < <(collect_model_roots "$rl_root")
  fi

  if [[ ${#model_roots[@]} -eq 0 ]]; then
    echo "[ERROR] no candidate model roots found."
    echo "        SFT_ROOT=$sft_root"
    echo "        RL_ROOT=$rl_root"
    return 1
  fi

  echo "[INFO] repo_root=$REPO_ROOT"
  echo "[INFO] eval_script=$eval_script"
  echo "[INFO] include_sft=$include_sft sft_root=$sft_root"
  echo "[INFO] include_rl=$include_rl rl_root=$rl_root"
  echo "[INFO] data_root=$data_root auto_data_mapping=$auto_data_mapping"
  echo "[INFO] model_filter=${model_filter:-<empty>}"
  echo "[INFO] force_reeval=$force_reeval dry_run=$dry_run"
  echo

  local total_models=0
  local scheduled_models=0
  local scheduled_ckpts=0
  local model_root model_name ckpt_list_str

  for model_root in "${model_roots[@]}"; do
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
      return 1
    fi
    if [[ ! -f "$RES_INDEX_PATH" ]]; then
      echo "[ERROR] index_path not found for model=$model_name: $RES_INDEX_PATH"
      return 1
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

    if [[ "$dry_run" == "1" ]]; then
      continue
    fi

    CATEGORY="$RES_CATEGORY" \
    TEST_DATA_PATH="$RES_TEST_DATA_PATH" \
    INDEX_PATH="$RES_INDEX_PATH" \
    CUDA_LIST="$cuda_list" \
    PYTHON_BIN="$python_bin" \
    CKPT_LIST="$ckpt_list_str" \
    bash "$eval_script" "$model_root"

    echo "[DONE] model=$model_name"
    echo
  done

  echo
  echo "[SUMMARY] total_models=$total_models scheduled_models=$scheduled_models scheduled_ckpts=$scheduled_ckpts"
  if [[ "$dry_run" == "1" ]]; then
    echo "[SUMMARY] dry run only, nothing executed."
  fi
}

if [[ $# -eq 0 ]]; then
  run_legacy_once
  exit $?
fi

COMMAND="$1"
shift

case "$COMMAND" in
  once)
    run_legacy_once "$@"
    ;;

  run|start|stop|status|tail)
    require_sidecar
    mkdir -p "$MANAGER_DIR"

    INSTANCE="${INSTANCE:-default}"
    FORWARD_ARGS=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --instance)
          INSTANCE="$2"
          shift 2
          ;;
        --instance=*)
          INSTANCE="${1#*=}"
          shift
          ;;
        *)
          FORWARD_ARGS+=("$1")
          shift
          ;;
      esac
    done

    INSTANCE_SAFE="$(echo "$INSTANCE" | tr '/ ' '__')"
    PID_FILE="$MANAGER_DIR/${INSTANCE_SAFE}.pid"
    LOG_FILE="$MANAGER_DIR/${INSTANCE_SAFE}.log"

    case "$COMMAND" in
      run)
        exec "$PYTHON_BIN" "$SIDECAR_PY" watch "${FORWARD_ARGS[@]}"
        ;;

      start)
        if is_running "$PID_FILE"; then
          echo "[INFO] watcher already running. instance=$INSTANCE pid=$(cat "$PID_FILE")"
          echo "[INFO] log=$LOG_FILE"
          exit 0
        fi
        nohup "$PYTHON_BIN" "$SIDECAR_PY" watch "${FORWARD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
        PID=$!
        echo "$PID" > "$PID_FILE"
        echo "[INFO] watcher started. instance=$INSTANCE pid=$PID"
        echo "[INFO] log=$LOG_FILE"
        ;;

      stop)
        if ! is_running "$PID_FILE"; then
          echo "[INFO] watcher not running. instance=$INSTANCE"
          rm -f "$PID_FILE"
          exit 0
        fi
        PID="$(cat "$PID_FILE")"
        kill "$PID" 2>/dev/null || true
        for _ in {1..20}; do
          if ! kill -0 "$PID" 2>/dev/null; then
            break
          fi
          sleep 0.2
        done
        if kill -0 "$PID" 2>/dev/null; then
          echo "[WARN] process still alive, sending SIGKILL pid=$PID"
          kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "[INFO] watcher stopped. instance=$INSTANCE"
        ;;

      status)
        if is_running "$PID_FILE"; then
          echo "[INFO] watcher running. instance=$INSTANCE pid=$(cat "$PID_FILE")"
          echo "[INFO] log=$LOG_FILE"
          exit 0
        fi
        echo "[INFO] watcher not running. instance=$INSTANCE"
        echo "[INFO] expected pid file: $PID_FILE"
        exit 1
        ;;

      tail)
        touch "$LOG_FILE"
        echo "[INFO] tailing log: $LOG_FILE"
        tail -n 100 -f "$LOG_FILE"
        ;;
    esac
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "[ERROR] unknown command: $COMMAND"
    usage
    exit 1
    ;;
esac
