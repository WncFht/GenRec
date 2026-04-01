#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_REPO_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_GENREC_REPO_DIR_DEFAULT="/Users/fanghaotian/Desktop/src/GenRec"
REMOTE_GENREC_REPO_DIR_DEFAULT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec"

LOCAL_GENREC_REPO_DIR="${LOCAL_GENREC_REPO_DIR:-$LOCAL_GENREC_REPO_DIR_DEFAULT}"
REMOTE_GENREC_REPO_DIR="${REMOTE_GENREC_REPO_DIR:-$REMOTE_GENREC_REPO_DIR_DEFAULT}"
CURRENT_REPO_ROOT="${CURRENT_REPO_ROOT:-$CURRENT_REPO_ROOT_DEFAULT}"
SOURCE_PROFILE="${SOURCE_PROFILE:-auto}"
DEST_PROFILE="${DEST_PROFILE:-auto}"
SOURCE_REPO_ROOT="${SOURCE_REPO_ROOT:-}"
DEST_REPO_ROOT="${DEST_REPO_ROOT:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUNDLE_NAME="${BUNDLE_NAME:-genrec_repo_bundle}"
CHUNK_SIZE="${CHUNK_SIZE:-10485760}"  # 10MB
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/sync_repo_bundle.sh pack [archive_path] <path> [<path> ...]
  bash scripts/sync_repo_bundle.sh unpack [archive_path|archive.part.000] [dest_root]
  bash scripts/sync_repo_bundle.sh list [archive_path|archive.part.000]

Description:
  - pack:
      Bundle one or more repo-relative or absolute paths under a GenRec root.
      Paths are stored relative to the source repo root, so unpack can restore
      them under the destination GenRec root automatically.
  - unpack:
      Restore bundled paths into the destination GenRec root. Existing target
      files/directories for bundled paths are replaced.
  - list:
      Print bundled relative paths without restoring them.

Root configuration:
  LOCAL_GENREC_REPO_DIR
  REMOTE_GENREC_REPO_DIR
  CURRENT_REPO_ROOT
  SOURCE_PROFILE=auto|local|remote
  DEST_PROFILE=auto|local|remote
  SOURCE_REPO_ROOT=/abs/path/to/source/root
  DEST_REPO_ROOT=/abs/path/to/dest/root

Other overrides:
  PYTHON_BIN
  BUNDLE_NAME
  CHUNK_SIZE
  KEEP_ARCHIVE=1   # keep source archive/parts after unpack

Examples:
  bash scripts/sync_repo_bundle.sh pack results_sync.tar.gz \
    log/train.log results/MyRun/checkpoint-100

  bash scripts/sync_repo_bundle.sh unpack results_sync.tar.gz

  DEST_PROFILE=remote bash scripts/sync_repo_bundle.sh unpack results_sync.tar.gz.part.000
USAGE
}

ensure_python() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: PYTHON_BIN not found: $PYTHON_BIN" >&2
    exit 1
  fi
}

resolve_profile_root() {
  local profile="$1"
  case "$profile" in
    auto)
      printf '%s\n' "$CURRENT_REPO_ROOT"
      ;;
    local)
      printf '%s\n' "$LOCAL_GENREC_REPO_DIR"
      ;;
    remote)
      printf '%s\n' "$REMOTE_GENREC_REPO_DIR"
      ;;
    *)
      echo "Error: unknown profile '$profile' (expected auto|local|remote)" >&2
      exit 1
      ;;
  esac
}

resolve_source_root() {
  if [[ -n "$SOURCE_REPO_ROOT" ]]; then
    printf '%s\n' "$SOURCE_REPO_ROOT"
  else
    resolve_profile_root "$SOURCE_PROFILE"
  fi
}

resolve_dest_root() {
  if [[ -n "$DEST_REPO_ROOT" ]]; then
    printf '%s\n' "$DEST_REPO_ROOT"
  else
    resolve_profile_root "$DEST_PROFILE"
  fi
}

normalize_archive_path() {
  local archive_arg="$1"
  local archive_path
  if [[ -z "$archive_arg" ]]; then
    archive_path="$(pwd)/${BUNDLE_NAME}_$(date +%Y%m%d_%H%M%S).tar.gz"
  elif [[ "$archive_arg" =~ \.part\.[0-9]{3}$ ]]; then
    archive_path="$archive_arg"
  elif [[ "$archive_arg" = /* ]]; then
    archive_path="$archive_arg"
  else
    archive_path="$(pwd)/$archive_arg"
  fi

  if [[ ! "$archive_path" =~ \.part\.[0-9]{3}$ && "$archive_path" != *.tar.gz ]]; then
    archive_path="${archive_path}.tar.gz"
  fi

  printf '%s\n' "$archive_path"
}

path_exists_under_root() {
  local root="$1"
  local path_arg="$2"

  "$PYTHON_BIN" - "$root" "$path_arg" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve()
path_arg = Path(sys.argv[2]).expanduser()
path = (path_arg if path_arg.is_absolute() else root / path_arg).resolve()

try:
    path.relative_to(root)
except ValueError:
    sys.exit(1)

sys.exit(0 if path.exists() else 1)
PY
}

archive_part_prefix() {
  local archive_path="$1"
  printf '%s.part.' "$archive_path"
}

find_latest_bundle_archive() {
  local search_dir="$1"
  local latest_archive
  latest_archive="$(ls -1t "$search_dir"/*.tar.gz "$search_dir"/*.tar.gz.part.000 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_archive" ]]; then
    return 1
  fi
  printf '%s\n' "$latest_archive"
}

split_archive_if_needed() {
  local archive_path="$1"
  local chunk_size="$2"
  local archive_size
  archive_size=$(wc -c < "$archive_path" | tr -d '[:space:]')
  if (( archive_size <= chunk_size )); then
    return
  fi

  local part_prefix
  part_prefix="$(archive_part_prefix "$archive_path")"
  rm -f "${part_prefix}"[0-9][0-9][0-9]
  split -b "$chunk_size" -d -a 3 "$archive_path" "$part_prefix"
  rm -f "$archive_path"

  echo "Archive exceeded CHUNK_SIZE=${chunk_size} bytes and was split into parts:"
  ls -1 "${part_prefix}"[0-9][0-9][0-9]
}

rebuild_archive_from_parts() {
  local input_path="$1"
  local output_path="$2"
  local part_prefix
  local -a part_files=()

  if [[ "$input_path" =~ \.part\.[0-9]{3}$ ]]; then
    part_prefix="${input_path%.part.[0-9][0-9][0-9]}.part."
  else
    part_prefix="$(archive_part_prefix "$input_path")"
  fi

  shopt -s nullglob
  part_files=("${part_prefix}"[0-9][0-9][0-9])
  shopt -u nullglob

  if [[ ${#part_files[@]} -eq 0 ]]; then
    echo "Error: no archive parts found for: $input_path" >&2
    exit 1
  fi

  cat "${part_files[@]}" > "$output_path"
}

resolve_relative_path() {
  local root="$1"
  local path_arg="$2"

  "$PYTHON_BIN" - "$root" "$path_arg" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve()
path_arg = Path(sys.argv[2]).expanduser()
path = (path_arg if path_arg.is_absolute() else root / path_arg).resolve()

try:
    rel = path.relative_to(root)
except ValueError:
    print(f"Error: path is outside source root: {path}", file=sys.stderr)
    sys.exit(2)

print(rel.as_posix())
PY
}

manifest_write() {
  local manifest_path="$1"
  local source_root="$2"
  shift 2
  "$PYTHON_BIN" - "$manifest_path" "$source_root" "$@" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
source_root = Path(sys.argv[2])
paths = []
for rel in sys.argv[3:]:
    rel_path = Path(rel)
    kind = "dir" if (source_root / rel_path).is_dir() else "file"
    paths.append({"path": rel_path.as_posix(), "type": kind})

payload = {
    "bundle_name": manifest_path.parent.name,
    "source_root": str(source_root),
    "paths": paths,
}

manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

manifest_list_paths() {
  local manifest_path="$1"
  "$PYTHON_BIN" - "$manifest_path" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for entry in manifest["paths"]:
    print(entry["path"])
PY
}

copy_to_stage() {
  local source_root="$1"
  local rel="$2"
  local payload_root="$3"
  local src="$source_root/$rel"
  local dest="$payload_root/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -a "$src" "$dest"
}

restore_from_stage() {
  local payload_root="$1"
  local rel="$2"
  local dest_root="$3"
  local src="$payload_root/$rel"
  local dest="$dest_root/$rel"
  mkdir -p "$(dirname "$dest")"
  rm -rf "$dest"
  cp -a "$src" "$dest"
}

pack_bundle() {
  if [[ $# -lt 1 ]]; then
    echo "Error: pack requires at least one file or directory." >&2
    usage
    exit 1
  fi

  ensure_python
  if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || (( CHUNK_SIZE <= 0 )); then
    echo "Error: CHUNK_SIZE must be a positive integer (bytes), got: $CHUNK_SIZE" >&2
    exit 1
  fi

  local source_root
  source_root="$(resolve_source_root)"
  if [[ ! -d "$source_root" ]]; then
    echo "Error: source repo root not found: $source_root" >&2
    exit 1
  fi

  local archive_arg=""
  if ! path_exists_under_root "$source_root" "$1"; then
    archive_arg="$1"
    shift
  fi

  if [[ $# -lt 1 ]]; then
    echo "Error: pack requires at least one source path after archive resolution." >&2
    usage
    exit 1
  fi

  local archive_path
  archive_path="$(normalize_archive_path "$archive_arg")"

  local stage_root bundle_root payload_root
  stage_root="$(mktemp -d)"
  bundle_root="$stage_root/$BUNDLE_NAME"
  payload_root="$bundle_root/payload"
  mkdir -p "$payload_root"

  local -a rel_paths=()
  local rel
  for path_arg in "$@"; do
    rel="$(resolve_relative_path "$source_root" "$path_arg")"
    if [[ ! -e "$source_root/$rel" ]]; then
      echo "Error: source path not found after normalization: $source_root/$rel" >&2
      rm -rf "$stage_root"
      exit 1
    fi
    rel_paths+=("$rel")
    copy_to_stage "$source_root" "$rel" "$payload_root"
  done

  manifest_write "$bundle_root/bundle_manifest.json" "$source_root" "${rel_paths[@]}"

  mkdir -p "$(dirname "$archive_path")"
  rm -f "$archive_path" "$(archive_part_prefix "$archive_path")"[0-9][0-9][0-9]
  tar -czf "$archive_path" -C "$stage_root" "$BUNDLE_NAME"
  split_archive_if_needed "$archive_path" "$CHUNK_SIZE"

  echo "Created repo bundle from: $source_root"
  echo "Bundle root inside archive: $BUNDLE_NAME"
  echo "Included paths:"
  printf '  %s\n' "${rel_paths[@]}"

  rm -rf "$stage_root"
}

extract_bundle_to_temp() {
  local archive_path="$1"
  local temp_dir="$2"
  tar -xzf "$archive_path" -C "$temp_dir"
}

resolve_archive_input() {
  local archive_arg="$1"
  local archive_path
  if [[ -n "$archive_arg" ]]; then
    archive_path="$(normalize_archive_path "$archive_arg")"
  else
    archive_path="$(find_latest_bundle_archive "$(pwd)")"
  fi
  printf '%s\n' "$archive_path"
}

list_bundle() {
  ensure_python
  local archive_path
  archive_path="$(resolve_archive_input "${1:-}")"
  local tmp_archive_path=""

  if [[ ! -f "$archive_path" ]]; then
    if [[ "$archive_path" =~ \.part\.[0-9]{3}$ || -f "$(archive_part_prefix "$archive_path")000" ]]; then
      :
    else
      echo "Error: archive not found: $archive_path" >&2
      exit 1
    fi
  fi

  if [[ "$archive_path" =~ \.part\.[0-9]{3}$ || ! -f "$archive_path" ]]; then
    tmp_archive_path="$(mktemp "${TMPDIR:-/tmp}/sync_repo_bundle_XXXXXX.tar.gz")"
    rebuild_archive_from_parts "$archive_path" "$tmp_archive_path"
    archive_path="$tmp_archive_path"
  fi

  local temp_extract_dir bundle_dir_name manifest_path
  temp_extract_dir="$(mktemp -d)"
  extract_bundle_to_temp "$archive_path" "$temp_extract_dir"
  bundle_dir_name="$(find "$temp_extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1 | xargs basename)"
  manifest_path="$temp_extract_dir/$bundle_dir_name/bundle_manifest.json"
  if [[ ! -f "$manifest_path" ]]; then
    echo "Error: bundle manifest missing: $manifest_path" >&2
    rm -rf "$temp_extract_dir"
    [[ -n "$tmp_archive_path" ]] && rm -f "$tmp_archive_path"
    exit 1
  fi

  echo "Archive: $archive_path"
  echo "Manifest: $manifest_path"
  manifest_list_paths "$manifest_path"

  rm -rf "$temp_extract_dir"
  if [[ -n "$tmp_archive_path" ]]; then
    rm -f "$tmp_archive_path"
  fi
}

unpack_bundle() {
  if [[ $# -gt 2 ]]; then
    echo "Error: unpack accepts at most [archive] [dest_root]." >&2
    usage
    exit 1
  fi

  ensure_python
  local archive_path
  archive_path="$(resolve_archive_input "${1:-}")"
  local requested_archive_path="$archive_path"
  if [[ $# -ge 2 && -n "${2:-}" ]]; then
    DEST_REPO_ROOT="$2"
  fi
  local dest_root
  dest_root="$(resolve_dest_root)"
  mkdir -p "$dest_root"

  local tmp_archive_path=""
  local -a source_archives=()
  if [[ ! -f "$archive_path" ]]; then
    if [[ "$archive_path" =~ \.part\.[0-9]{3}$ || -f "$(archive_part_prefix "$archive_path")000" ]]; then
      :
    else
      echo "Error: archive not found: $archive_path" >&2
      exit 1
    fi
  fi

  if [[ "$archive_path" =~ \.part\.[0-9]{3}$ || ! -f "$archive_path" ]]; then
    local part_prefix
    if [[ "$requested_archive_path" =~ \.part\.[0-9]{3}$ ]]; then
      part_prefix="${requested_archive_path%.part.[0-9][0-9][0-9]}.part."
    else
      part_prefix="$(archive_part_prefix "$requested_archive_path")"
    fi
    shopt -s nullglob
    source_archives=("${part_prefix}"[0-9][0-9][0-9])
    shopt -u nullglob
    tmp_archive_path="$(mktemp "${TMPDIR:-/tmp}/sync_repo_bundle_XXXXXX.tar.gz")"
    rebuild_archive_from_parts "$archive_path" "$tmp_archive_path"
    archive_path="$tmp_archive_path"
  else
    source_archives=("$archive_path")
  fi

  local temp_extract_dir bundle_dir_name bundle_root manifest_path payload_root
  temp_extract_dir="$(mktemp -d)"
  extract_bundle_to_temp "$archive_path" "$temp_extract_dir"
  bundle_dir_name="$(find "$temp_extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1 | xargs basename)"
  if [[ -z "$bundle_dir_name" ]]; then
    echo "Error: failed to detect bundle directory in archive: $archive_path" >&2
    rm -rf "$temp_extract_dir"
    [[ -n "$tmp_archive_path" ]] && rm -f "$tmp_archive_path"
    exit 1
  fi

  bundle_root="$temp_extract_dir/$bundle_dir_name"
  manifest_path="$bundle_root/bundle_manifest.json"
  payload_root="$bundle_root/payload"
  if [[ ! -f "$manifest_path" ]]; then
    echo "Error: bundle manifest missing after extraction: $manifest_path" >&2
    rm -rf "$temp_extract_dir"
    [[ -n "$tmp_archive_path" ]] && rm -f "$tmp_archive_path"
    exit 1
  fi

  local -a rel_paths=()
  local rel
  while IFS= read -r rel; do
    [[ -n "$rel" ]] && rel_paths+=("$rel")
  done < <(manifest_list_paths "$manifest_path")
  if [[ ${#rel_paths[@]} -eq 0 ]]; then
    echo "Error: bundle manifest contains no paths: $manifest_path" >&2
    rm -rf "$temp_extract_dir"
    [[ -n "$tmp_archive_path" ]] && rm -f "$tmp_archive_path"
    exit 1
  fi

  for rel in "${rel_paths[@]}"; do
    if [[ ! -e "$payload_root/$rel" ]]; then
      echo "Error: bundled path missing from payload: $payload_root/$rel" >&2
      rm -rf "$temp_extract_dir"
      [[ -n "$tmp_archive_path" ]] && rm -f "$tmp_archive_path"
      exit 1
    fi
    restore_from_stage "$payload_root" "$rel" "$dest_root"
    echo "Restored: $dest_root/$rel"
  done

  echo "Unpacked repo bundle into: $dest_root"

  rm -rf "$temp_extract_dir"
  if [[ -n "$tmp_archive_path" ]]; then
    rm -f "$tmp_archive_path"
  fi
  if [[ "$KEEP_ARCHIVE" != "1" ]]; then
    local archive_file
    for archive_file in "${source_archives[@]}"; do
      [[ -f "$archive_file" ]] && rm -f "$archive_file"
    done
  fi
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    pack)
      shift
      pack_bundle "$@"
      ;;
    unpack)
      shift
      unpack_bundle "$@"
      ;;
    list)
      shift
      list_bundle "${1:-}"
      ;;
    -h|--help|help|'')
      usage
      ;;
    *)
      echo "Error: unknown command: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
