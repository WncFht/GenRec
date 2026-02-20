#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SIDECAR_PY="${SIDECAR_PY:-$SCRIPT_DIR/eval_wandb_sidecar.py}"
MANAGER_DIR="${MANAGER_DIR:-$SCRIPT_DIR/log/eval_sidecar}"

usage() {
  cat <<'EOF'
Usage:
  bash eval_wandb_sidecar.sh <start|stop|status|tail|run|once> [--instance NAME] [sidecar args...]

Examples:
  bash eval_wandb_sidecar.sh start --instance cb64 --sft-root /path/to/saves/model --category Instruments_grec --test-data-path /path/to/test.json --index-path /path/to/id2sid.json --cuda-list "0"
  bash eval_wandb_sidecar.sh status --instance cb64
  bash eval_wandb_sidecar.sh tail --instance cb64
  bash eval_wandb_sidecar.sh stop --instance cb64

Notes:
  - Additional args are passed directly to eval_wandb_sidecar.py.
  - "run" runs in foreground (watch mode).
  - "once" runs one pass then exits.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

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

mkdir -p "$MANAGER_DIR"
INSTANCE_SAFE="$(echo "$INSTANCE" | tr '/ ' '__')"
PID_FILE="$MANAGER_DIR/${INSTANCE_SAFE}.pid"
LOG_FILE="$MANAGER_DIR/${INSTANCE_SAFE}.log"

is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  kill -0 "$pid" 2>/dev/null
}

require_sidecar() {
  if [[ ! -f "$SIDECAR_PY" ]]; then
    echo "[ERROR] sidecar script not found: $SIDECAR_PY"
    exit 1
  fi
}

case "$COMMAND" in
  start)
    require_sidecar
    if is_running; then
      echo "[INFO] sidecar already running. instance=$INSTANCE pid=$(cat "$PID_FILE")"
      echo "[INFO] log=$LOG_FILE"
      exit 0
    fi

    nohup "$PYTHON_BIN" "$SIDECAR_PY" "${FORWARD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"
    echo "[INFO] sidecar started. instance=$INSTANCE pid=$PID"
    echo "[INFO] log=$LOG_FILE"
    ;;

  run)
    require_sidecar
    exec "$PYTHON_BIN" "$SIDECAR_PY" "${FORWARD_ARGS[@]}"
    ;;

  once)
    require_sidecar
    exec "$PYTHON_BIN" "$SIDECAR_PY" --once "${FORWARD_ARGS[@]}"
    ;;

  stop)
    if ! is_running; then
      echo "[INFO] sidecar not running. instance=$INSTANCE"
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
    echo "[INFO] sidecar stopped. instance=$INSTANCE"
    ;;

  status)
    if is_running; then
      echo "[INFO] sidecar running. instance=$INSTANCE pid=$(cat "$PID_FILE")"
      echo "[INFO] log=$LOG_FILE"
      exit 0
    fi
    echo "[INFO] sidecar not running. instance=$INSTANCE"
    echo "[INFO] expected pid file: $PID_FILE"
    exit 1
    ;;

  tail)
    touch "$LOG_FILE"
    echo "[INFO] tailing log: $LOG_FILE"
    tail -n 100 -f "$LOG_FILE"
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
