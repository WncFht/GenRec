#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
SFT_ROOT="${1:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/GenRec/saves/qwen2.5-3b/full/Industrial_and_Scientific-sft-dsz0-4gpu-eq8}"
CUDA_LIST="${CUDA_LIST:-0 1 2 3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TEST_DATA_PATH="${TEST_DATA_PATH:-data/${CATEGORY}/sft/test.json}"
INDEX_PATH="${INDEX_PATH:-data/${CATEGORY}/id2sid.json}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_BEAMS="${NUM_BEAMS:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DO_SAMPLE="${DO_SAMPLE:-False}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"
SID_LEVELS="${SID_LEVELS:--1}"  # <=0 means auto(all levels), supports both 3-level and 4-level SIDs
# manual ckpt list (optional): "checkpoint-495 checkpoint-630" or absolute paths.
# if empty, auto evaluate all checkpoint-* under SFT_ROOT.
CKPT_LIST="${CKPT_LIST:-}"

resolve_latest_checkpoint() {
  local input_path="$1"
  local resolved=""

  if [[ ! -d "$input_path" ]]; then
    echo "Error: SFT path not found: $input_path"
    exit 1
  fi

  if [[ "$(basename "$input_path")" == checkpoint-* ]]; then
    resolved="$input_path"
  else
    resolved="$(find "$input_path" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
  fi

  if [[ -z "$resolved" ]]; then
    echo "Error: No checkpoint-* found under: $input_path"
    exit 1
  fi

  echo "$resolved"
}

echo "=========================================="
echo "Industry Recommendation Evaluation (SFT)"
echo "SFT root: ${SFT_ROOT}"
if [[ -n "$CKPT_LIST" ]]; then
  echo "CKPT_LIST (manual): ${CKPT_LIST}"
else
  echo "CKPT_LIST (manual): <empty, auto all checkpoints>"
fi
echo "Test data: ${TEST_DATA_PATH}"
echo "CUDA list: ${CUDA_LIST}"
echo "=========================================="

if [[ ! -f "$TEST_DATA_PATH" ]]; then
  echo "Error: Test file not found: $TEST_DATA_PATH"
  exit 1
fi
if [[ ! -f "$INDEX_PATH" ]]; then
  echo "Error: Index file not found: $INDEX_PATH"
  exit 1
fi

read -r -a GPU_ARR <<< "$CUDA_LIST"

run_single_gpu_eval() {
  local model_path="$1"
  local temp_dir="$2"
  local output_dir="$3"
  local gpu_id="$4"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u evaluate.py \
    --model_name_or_path "$model_path" \
    --test_data_path "$TEST_DATA_PATH" \
    --result_json_path "$temp_dir/result.json" \
    --index_path "$INDEX_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_beams "$NUM_BEAMS" \
    --temperature "$TEMPERATURE" \
    --do_sample "$DO_SAMPLE" \
    --length_penalty "$LENGTH_PENALTY" \
    --sid_levels "$SID_LEVELS" \
    --compute_metrics_flag True

  cp "$temp_dir/result.json" "$output_dir/final_result.json"
  if [[ -f "$temp_dir/metrics.json" ]]; then
    cp "$temp_dir/metrics.json" "$output_dir/metrics.json"
  fi
  if [[ -f "$temp_dir/metrics.tsv" ]]; then
    cp "$temp_dir/metrics.tsv" "$output_dir/metrics.tsv"
  fi
}

collect_model_paths() {
  local root="$1"
  MODEL_PATHS=()

  if [[ -n "$CKPT_LIST" ]]; then
    local list="${CKPT_LIST//,/ }"
    local ckpt_input
    for ckpt_input in $list; do
      local candidate="$ckpt_input"
      if [[ "$candidate" != /* ]]; then
        candidate="${root}/${candidate}"
      fi
      MODEL_PATHS+=("$(resolve_latest_checkpoint "$candidate")")
    done
    return
  fi

  if [[ ! -d "$root" ]]; then
    echo "Error: SFT path not found: $root"
    exit 1
  fi
  if [[ "$(basename "$root")" == checkpoint-* ]]; then
    MODEL_PATHS+=("$root")
    return
  fi

  mapfile -t MODEL_PATHS < <(find "$root" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
    echo "Error: No checkpoint-* found under: $root"
    exit 1
  fi
}

evaluate_one_model() {
  local model_path="$1"
  local model_parent
  local model_basename
  local temp_dir
  local output_dir

  model_parent="$(basename "$(dirname "$model_path")")"
  model_basename="$(basename "$model_path")"
  temp_dir="temp/eval-${CATEGORY}-sft3b-${model_parent}-${model_basename}"
  output_dir="results/${model_parent}/${model_basename}"

  echo "------------------------------------------"
  echo "Evaluate checkpoint: ${model_path}"
  echo "Output dir: ${output_dir}"
  echo "------------------------------------------"

  rm -rf "$temp_dir"
  mkdir -p "$temp_dir" "$output_dir"

  if [[ ${#GPU_ARR[@]} -le 1 ]]; then
    local gpu_id="${GPU_ARR[0]:-0}"
    echo "Single GPU evaluation on GPU ${gpu_id} ..."
    run_single_gpu_eval "$model_path" "$temp_dir" "$output_dir" "$gpu_id"
    rm -rf "$temp_dir"
    echo "Done. Results: $output_dir"
    return
  fi

  if [[ ! -f "split_json.py" || ! -f "merge_json.py" ]]; then
    echo "Warning: split_json.py or merge_json.py not found, fallback to single GPU."
    run_single_gpu_eval "$model_path" "$temp_dir" "$output_dir" "${GPU_ARR[0]}"
    rm -rf "$temp_dir"
    echo "Done. Results: $output_dir"
    return
  fi

  echo "Multi-GPU evaluation..."
  "$PYTHON_BIN" split_json.py --input_path "$TEST_DATA_PATH" --output_path "$temp_dir" --cuda_list "$CUDA_LIST"

  local gpu_id
  for gpu_id in "${GPU_ARR[@]}"; do
    if [[ -f "$temp_dir/${gpu_id}.json" ]]; then
      echo "Start GPU ${gpu_id} ..."
      CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u evaluate.py \
        --model_name_or_path "$model_path" \
        --test_data_path "$temp_dir/${gpu_id}.json" \
        --result_json_path "$temp_dir/${gpu_id}_result.json" \
        --index_path "$INDEX_PATH" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_beams "$NUM_BEAMS" \
        --temperature "$TEMPERATURE" \
        --do_sample "$DO_SAMPLE" \
        --length_penalty "$LENGTH_PENALTY" \
        --sid_levels "$SID_LEVELS" \
        --compute_metrics_flag False &
    fi
  done

  wait

  "$PYTHON_BIN" merge_json.py --input_path "$temp_dir" --output_path "$output_dir/final_result.json" --cuda_list "$CUDA_LIST" --compute_metrics False
  "$PYTHON_BIN" -u evaluate.py --result_json_path "$output_dir/final_result.json" --metrics_only True

  if [[ -f "$output_dir/metrics.json" ]]; then
    echo "Metrics: $output_dir/metrics.json"
  fi
  if [[ -f "$output_dir/metrics.tsv" ]]; then
    echo "Metrics TSV: $output_dir/metrics.tsv"
  fi

  rm -rf "$temp_dir"
  echo "Done. Results: $output_dir"
}

collect_model_paths "$SFT_ROOT"
echo "Will evaluate ${#MODEL_PATHS[@]} checkpoint(s)."
printf ' - %s\n' "${MODEL_PATHS[@]}"

for model_path in "${MODEL_PATHS[@]}"; do
  evaluate_one_model "$model_path"
done

echo "All evaluations finished."
