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

resolve_model_path() {
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

MODEL_PATH="$(resolve_model_path "$SFT_ROOT")"
MODEL_PARENT="$(basename "$(dirname "$MODEL_PATH")")"
MODEL_BASENAME="$(basename "$MODEL_PATH")"

TEMP_DIR="temp/eval-${CATEGORY}-sft3b"
OUTPUT_DIR="results/${MODEL_PARENT}/${MODEL_BASENAME}"

echo "=========================================="
echo "Industry Recommendation Evaluation (SFT)"
echo "SFT root: ${SFT_ROOT}"
echo "Model checkpoint: ${MODEL_PATH}"
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

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR" "$OUTPUT_DIR"

read -r -a GPU_ARR <<< "$CUDA_LIST"

run_single_gpu_eval() {
  local gpu_id="$1"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u evaluate.py \
    --model_name_or_path "$MODEL_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --result_json_path "$TEMP_DIR/result.json" \
    --index_path "$INDEX_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_beams "$NUM_BEAMS" \
    --temperature "$TEMPERATURE" \
    --do_sample "$DO_SAMPLE" \
    --length_penalty "$LENGTH_PENALTY" \
    --sid_levels "$SID_LEVELS" \
    --compute_metrics_flag True

  cp "$TEMP_DIR/result.json" "$OUTPUT_DIR/final_result.json"
  if [[ -f "$TEMP_DIR/metrics.json" ]]; then
    cp "$TEMP_DIR/metrics.json" "$OUTPUT_DIR/metrics.json"
  fi
  if [[ -f "$TEMP_DIR/metrics.tsv" ]]; then
    cp "$TEMP_DIR/metrics.tsv" "$OUTPUT_DIR/metrics.tsv"
  fi
}

if [[ ${#GPU_ARR[@]} -le 1 ]]; then
  GPU_ID="${GPU_ARR[0]:-0}"
  echo "Single GPU evaluation on GPU ${GPU_ID} ..."
  run_single_gpu_eval "$GPU_ID"
  rm -rf "$TEMP_DIR"
  echo "Done. Results: $OUTPUT_DIR"
  exit 0
fi

if [[ ! -f "split_json.py" || ! -f "merge_json.py" ]]; then
  echo "Warning: split_json.py or merge_json.py not found, fallback to single GPU."
  run_single_gpu_eval "${GPU_ARR[0]}"
  rm -rf "$TEMP_DIR"
  echo "Done. Results: $OUTPUT_DIR"
  exit 0
fi

echo "Multi-GPU evaluation..."
"$PYTHON_BIN" split_json.py --input_path "$TEST_DATA_PATH" --output_path "$TEMP_DIR" --cuda_list "$CUDA_LIST"

for gpu_id in "${GPU_ARR[@]}"; do
  if [[ -f "$TEMP_DIR/${gpu_id}.json" ]]; then
    echo "Start GPU ${gpu_id} ..."
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u evaluate.py \
      --model_name_or_path "$MODEL_PATH" \
      --test_data_path "$TEMP_DIR/${gpu_id}.json" \
      --result_json_path "$TEMP_DIR/${gpu_id}_result.json" \
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

"$PYTHON_BIN" merge_json.py --input_path "$TEMP_DIR" --output_path "$OUTPUT_DIR/final_result.json" --cuda_list "$CUDA_LIST" --compute_metrics False
"$PYTHON_BIN" -u evaluate.py --result_json_path "$OUTPUT_DIR/final_result.json" --metrics_only True

if [[ -f "$OUTPUT_DIR/metrics.json" ]]; then
  echo "Metrics: $OUTPUT_DIR/metrics.json"
fi
if [[ -f "$OUTPUT_DIR/metrics.tsv" ]]; then
  echo "Metrics TSV: $OUTPUT_DIR/metrics.tsv"
fi

rm -rf "$TEMP_DIR"
echo "Done. Results: $OUTPUT_DIR"
