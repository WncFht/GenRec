#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl.sh [options]

Run modes:
  --nohup                 Start RL in background via nohup and follow log (default)
  --detach                Start RL in background via nohup and do not follow log
  --tail                  Follow latest log for current run name
  --run                   Internal mode; run training command directly

Config selection:
  --preset <name>         Preset config shortcut (default: prefix_token)
  --config <path>         Explicit config path; overrides --preset

Runtime / launch overrides:
  --run-name <name>
  --wandb-mode <offline|online|disabled>
  --num-processes <n>
  --port <n>
  --ds-config <path>
  --conda-activate <path>
  --conda-env <name>
  --log-dir <path>
  --log-file <path>
  --dry-run
  -h, --help

Trainer overrides:
  Common `trl_trainer.py` options such as `--output-dir`, `--eval_on_start`,
  `--set section.key=value`, `--print-config`, and `--print-flat-kwargs`
  are passed through to `trl_trainer.py`.
EOF
}

sanitize_log_name() {
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
  latest="$(ls -1t "${log_dir}/$(sanitize_log_name "${prefix}")"_*.log 2>/dev/null | head -n 1 || true)"
  echo "$latest"
}

resolve_preset_config_path() {
  local preset="$1"
  preset="$(echo "$preset" | tr '[:upper:]' '[:lower:]')"
  case "$preset" in
    prefix_token|default)
      echo "${REPO_ROOT}/configs/rl/instruments/prefix_token.json"
      ;;
    prefix_token_totalnorm)
      echo "${REPO_ROOT}/configs/rl/instruments/prefix_token_totalnorm.json"
      ;;
    prefix_token_totalnorm_errtok)
      echo "${REPO_ROOT}/configs/rl/instruments/prefix_token_totalnorm_errtok.json"
      ;;
    prefix_token_only)
      echo "${REPO_ROOT}/configs/rl/instruments/prefix_token_only.json"
      ;;
    prefix_seq_only)
      echo "${REPO_ROOT}/configs/rl/instruments/prefix_seq_only.json"
      ;;
    rule_only)
      echo "${REPO_ROOT}/configs/rl/instruments/rule_only.json"
      ;;
    ranking)
      echo "${REPO_ROOT}/configs/rl/instruments/ranking.json"
      ;;
    ranking_only)
      echo "${REPO_ROOT}/configs/rl/instruments/ranking_only.json"
      ;;
    *)
      echo "[ERROR] Unknown preset: $preset" >&2
      echo "[ERROR] Supported presets: prefix_token, prefix_token_totalnorm, prefix_token_totalnorm_errtok, prefix_token_only, prefix_seq_only, rule_only, ranking, ranking_only" >&2
      exit 1
      ;;
  esac
}

resolve_preset_default_run_name() {
  local preset="$1"
  preset="$(echo "$preset" | tr '[:upper:]' '[:lower:]')"
  case "$preset" in
    prefix_token|default)
      echo "instruments_grec_rl_prefix_tokenadv_ndcg_rule0_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
    prefix_token_totalnorm)
      echo "instruments_grec_rl_prefix_tokenadv_totalnorm_ndcg_rule0_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
    prefix_token_totalnorm_errtok)
      echo "instruments_grec_rl_prefix_tokenadv_totalnorm_errtok_ndcg_rule0_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
    prefix_token_only)
      echo "instruments_grec_rl_prefix_token_only_totalnorm_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
    prefix_seq_only)
      echo "instruments_grec_rl_prefix_seq_only_fixbool_rerun_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
    rule_only)
      echo "instruments_grec_rl_rule_only_rerun_quietlog_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
    ranking)
      echo "instruments_grec_rl_ranking_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
    ranking_only)
      echo "instruments_grec_rl_ranking_only_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
    *)
      echo "instruments_grec_rl_qwen2_5_3b_qwen4b_4_256_from_ckpt495"
      ;;
  esac
}

trainer_option_is_flag() {
  case "$1" in
    --print-config|--print-flat-kwargs)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

trainer_option_expects_value() {
  case "$1" in
    --set|--model|--data-dir|--data_dir|--index-path|--index_path|--output-dir|--output_dir|--prefix|--num-beams|--num_beams|--sid-levels|--sid_levels|--reward-mode|--reward_mode|--hint-mode|--fixed-hint-depth-map-path|--fixed_hint_depth_map_path|--fixed-hint-depth-cap|--fixed_hint_depth_cap|--fixed-hint-unsolved-depth|--fixed_hint_unsolved_depth|--fixed-hint-task-names|--fixed_hint_task_names|--fixed-hint-apply-to-eval|--fixed_hint_apply_to_eval|--dynamic-hint-max-depth|--dynamic_hint_max_depth|--dynamic-hint-apply-to-eval|--dynamic_hint_apply_to_eval|--dynamic-hint-task-names|--dynamic_hint_task_names|--hint-ce-loss-coef|--hint_ce_loss_coef|--token-level-prefix-advantage|--token_level_prefix_advantage|--token-adv-total-token-normalize|--token_adv_total_token_normalize|--token-level-ndcg-error-token-penalty|--token_level_ndcg_error_token_penalty|--prefix-reward-normalize|--prefix_reward_normalize|--probe-rule-with-zero-weight|--probe_rule_with_zero_weight|--temperature|--top-p|--top_p|--top-k|--top_k|--max-completion-length|--max_completion_length|--beta|--repetition-penalty|--repetition_penalty|--do-sample|--do_sample|--per-device-train-batch-size|--per_device_train_batch_size|--per-device-eval-batch-size|--per_device_eval_batch_size|--gradient-accumulation-steps|--gradient_accumulation_steps|--num-train-epochs|--num_train_epochs|--learning-rate|--learning_rate|--logging-steps|--logging_steps|--eval-step|--eval_step|--eval-strategy|--eval_strategy|--eval-on-start|--eval_on_start|--save-strategy|--save_strategy|--save-steps|--save_steps|--save-total-limit|--save_total_limit|--save-only-model|--save_only_model|--warmup-ratio|--warmup_ratio|--max-grad-norm|--max_grad_norm|--optim|--lr-scheduler-type|--lr_scheduler_type|--bf16|--deepspeed|--report-to|--report_to|--resume-from-checkpoint|--resume_from_checkpoint|--train-task-names|--train_task_names|--eval-task-names|--eval_task_names)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

setup_runtime_env() {
  if [[ ! -f "$CONDA_ACTIVATE" ]]; then
    echo "[ERROR] Conda activate script not found: $CONDA_ACTIVATE" >&2
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
}

run_preflight_checks() {
  require_exists "$REPO_ROOT" "REPO_ROOT"
  require_file "$CONFIG_PATH" "RL config"
  require_file "$DS_CONFIG" "DeepSpeed config"
  require_file "${REPO_ROOT}/trl_trainer.py" "trl_trainer.py"

  setup_runtime_env

  if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] accelerate not found in PATH" >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODE="nohup"
DRY_RUN=0
FROM_NOHUP=0
LOG_FILE_OVERRIDE=""
PASSTHROUGH_ARGS=()
CONFIG_FROM_PRESET=0
FORWARD_RUN_NAME=0
RUN_NAME_SET=0

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/conda/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-genrec}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
CONFIG_PATH="${CONFIG_PATH:-}"
PRESET="${PRESET:-prefix_token}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PORT="${MAIN_PORT:-29513}"
DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/config/zero2.yaml}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log}"

export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

if [[ -n "${RUN_NAME:-}" ]]; then
  RUN_NAME_SET=1
elif [[ -n "${WANDB_RUN_NAME:-}" ]]; then
  RUN_NAME="${WANDB_RUN_NAME}"
  RUN_NAME_SET=1
else
  RUN_NAME=""
fi

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
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run-name|--run_name)
      RUN_NAME="$2"
      RUN_NAME_SET=1
      shift 2
      ;;
    --wandb-mode)
      export WANDB_MODE="$2"
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
    --)
      shift
      while [[ $# -gt 0 ]]; do
        PASSTHROUGH_ARGS+=("$1")
        shift
      done
      break
      ;;
    --*=*)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      if trainer_option_is_flag "$1"; then
        PASSTHROUGH_ARGS+=("$1")
        shift
      elif trainer_option_expects_value "$1"; then
        if [[ $# -lt 2 ]]; then
          echo "[ERROR] Missing value for trainer option: $1" >&2
          exit 1
        fi
        PASSTHROUGH_ARGS+=("$1" "$2")
        shift 2
      else
        echo "[ERROR] Unknown option: $1" >&2
        exit 1
      fi
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$(resolve_preset_config_path "$PRESET")"
  CONFIG_FROM_PRESET=1
fi

if [[ "$RUN_NAME_SET" -eq 1 ]]; then
  FORWARD_RUN_NAME=1
elif [[ "$CONFIG_FROM_PRESET" -eq 1 ]]; then
  RUN_NAME="$(resolve_preset_default_run_name "$PRESET")"
  FORWARD_RUN_NAME=1
elif [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(sanitize_log_name "$(basename -- "${CONFIG_PATH%.*}")")"
fi

if [[ "$FORWARD_RUN_NAME" -eq 1 ]]; then
  export WANDB_RUN_NAME="$RUN_NAME"
fi

if [[ -n "$LOG_FILE_OVERRIDE" ]]; then
  LOG_FILE="$LOG_FILE_OVERRIDE"
else
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/$(sanitize_log_name "${RUN_NAME}")_${TS}.log"
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

build_train_cmd() {
  TRAIN_CMD=(
    accelerate launch
    --config_file "$DS_CONFIG"
    --num_processes "$NUM_PROCESSES"
    --main_process_port "$MAIN_PORT"
    trl_trainer.py
    --config "$CONFIG_PATH"
  )
  if [[ "$FORWARD_RUN_NAME" -eq 1 ]]; then
    TRAIN_CMD+=(--run_name "$RUN_NAME")
  fi
  if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    TRAIN_CMD+=("${PASSTHROUGH_ARGS[@]}")
  fi
}

build_train_cmd

if [[ "$MODE" == "nohup" || "$MODE" == "detach" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "${TRAIN_CMD[@]}"
    echo
    exit 0
  fi

  run_preflight_checks

  mkdir -p "$(dirname -- "$LOG_FILE")"
  touch "$LOG_FILE"

  CHILD_ARGS=(
    --run
    --from-nohup
    --log-file "$LOG_FILE"
    --num-processes "$NUM_PROCESSES"
    --port "$MAIN_PORT"
    --ds-config "$DS_CONFIG"
    --conda-activate "$CONDA_ACTIVATE"
    --conda-env "$CONDA_ENV_NAME"
  )
  if [[ "$CONFIG_FROM_PRESET" -eq 1 ]]; then
    CHILD_ARGS+=(--preset "$PRESET")
  else
    CHILD_ARGS+=(--config "$CONFIG_PATH")
  fi
  if [[ "$FORWARD_RUN_NAME" -eq 1 ]]; then
    CHILD_ARGS+=(--run-name "$RUN_NAME")
  fi
  if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    CHILD_ARGS+=("${PASSTHROUGH_ARGS[@]}")
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

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
  exit 0
fi

run_preflight_checks

cd "$REPO_ROOT"

mkdir -p "$(dirname -- "$LOG_FILE")"
if [[ "$FROM_NOHUP" -eq 0 ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "[INFO] CONFIG_PATH=$CONFIG_PATH"
echo "[INFO] DS_CONFIG=$DS_CONFIG"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] NUM_PROCESSES=$NUM_PROCESSES"
echo "[INFO] MAIN_PORT=$MAIN_PORT"
echo "[INFO] RUN_NAME=${RUN_NAME:-<config>}"
echo "[INFO] PRESET=$PRESET"

"${TRAIN_CMD[@]}"
