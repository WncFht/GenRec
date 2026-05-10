#!/usr/bin/env bash
set -eo pipefail

run_rl_variant_wrapper() {
  local script_dir default_repo_root repo_root base_script
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  default_repo_root="$(cd -- "${script_dir}/../.." && pwd)"
  repo_root="${REPO_ROOT:-$default_repo_root}"
  base_script="${script_dir}/Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl.sh"

  if [[ ! -f "$base_script" ]]; then
    echo "[ERROR] Missing base RL wrapper: $base_script" >&2
    exit 1
  fi

  REPO_ROOT="$repo_root" exec bash "$base_script" "$@"
}
