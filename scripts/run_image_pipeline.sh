#!/bin/bash
set -euo pipefail
SCRIPT_START_TIME=$(date +%s)

############################################
# Repo-relative paths
############################################

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
CONFIGS_DIR="$ROOT_DIR/configs"

############################################
# User config
############################################

VLM_MODEL="Mumulin/Qwen3-VL-4B-Instruct-NTUH"
VLM_TAG="$(basename "$VLM_MODEL")"

############################################
# HF repos
############################################

CLARITY_REPO="Mumulin/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k-NTUH"

NEED_REPOS=(
  "Mumulin/beit3_large_patch16_224.in22k_ft_in1k-NTUH"
  "Mumulin/beit3_base_patch16_224.indomain_pt-NTUH"
  "Mumulin/swinv2_large_window12_192-NTUH"
)

############################################
# Paths
############################################

INPUT_ROOT="$ROOT_DIR/data"  
OUTPUTS_ROOT="$ROOT_DIR/.work"
RESULTS_ROOT="$ROOT_DIR/results"

############################################
# Process all folders in input root
############################################

echo "📄 Processing all folders in $INPUT_ROOT"

# Get all directories in INPUT_ROOT
INPUT_DIRS=()
while IFS= read -r -d '' dir; do
    INPUT_DIRS+=("$dir")
done < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
    echo "❌ No directories found in $INPUT_ROOT"
    exit 1
fi

echo "📄 Found ${#INPUT_DIRS[@]} directories"

############################################
# Loop directories
############################################

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
  CASE_START_TIME=$(date +%s)

  # Extract directory name as case ID
  CASE_ID=$(basename "$INPUT_DIR")
  echo "▶️ PROCESSING: $CASE_ID"

  WORK_DIR="$OUTPUTS_ROOT/$CASE_ID"
  CSV_RUN_NEED="$WORK_DIR/run_csv_need"

  mkdir -p "$CSV_RUN_NEED"

  CLARITY_CSV="$WORK_DIR/clarity.csv"
  CLEAR_LIST="$WORK_DIR/clarity_filtered.txt"
  NEED_FUSED="$WORK_DIR/need_vote.csv"
  NEED_LIST="$WORK_DIR/need_list.txt"
  QA_JSON="$WORK_DIR/phase_query.json"

  ##########################################
  # A) CLARITY
  ##########################################

  python -m utils.run_frame_classifier \
    --data-dir "$INPUT_DIR" \
    --dataset ImageFolder \
    --model "$CLARITY_REPO" \
    --results-dir "$WORK_DIR" \
    --results-file "clarity.csv" \
    --batch-size 128 \
    --workers 32 \
    --topk 1 \
    --amp \
    --fullname

  awk -F',' '
    NR==1 { for(i=1;i<=NF;i++){ if($i=="filename") fn=i; if($i=="top1_label") lb=i } }
    NR>1 && $lb==1 { print $fn }
  ' "$CLARITY_CSV" > "$CLEAR_LIST"

  ##########################################
  # B) NEED
  ##########################################

  NEED_RUNS=()

  for REPO in "${NEED_REPOS[@]}"; do
    NAME="$(basename "$REPO")"
    OUT_CSV="$CSV_RUN_NEED/${NAME}.csv"

    python -m utils.run_frame_classifier \
      --data-dir "$INPUT_DIR" \
      --input-list "$CLEAR_LIST" \
      --dataset ImageFolder \
      --model "$REPO" \
      --results-dir "$CSV_RUN_NEED" \
      --results-file "$(basename "$OUT_CSV")" \
      --batch-size 128 \
      --workers 32 \
      --topk 1 \
      --amp \
      --fullname

    NEED_RUNS+=("$OUT_CSV")
  done

  ##########################################
  # C) FUSE
  ##########################################

  python -m utils.fuse_frame_votes \
    "$NEED_FUSED" 0.9 2 1 "${NEED_RUNS[@]}"

  python - <<PY
import pandas as pd
df = pd.read_csv("$NEED_FUSED")
df[df["final_label"] == 0]["filename"].to_csv(
    "$NEED_LIST", index=False, header=False
)
PY

  ##########################################
  # D) QA JSON
  ##########################################

  python -m utils.build_vlm_dataset \
    --input-list "$NEED_LIST" \
    --output-json "$QA_JSON" \
    --question-schema "$CONFIGS_DIR/image_phase_questions.json"

  ##########################################
  # E) VLM inference
  ##########################################

  VLM_JSONL="$WORK_DIR/${CASE_ID}.jsonl"

  python -m utils.run_vlm_inference \
    --model_name_or_path "$VLM_MODEL" \
    --input_json "$QA_JSON" \
    --output_jsonl "$VLM_JSONL" \
    --batch_size 16

  ##########################################
  # F) FINAL OUTPUT
  ##########################################

  OUT_CSV="$RESULTS_ROOT/$VLM_TAG/${CASE_ID}_phase_query.csv"
  mkdir -p "$(dirname "$OUT_CSV")"

  python -m utils.image_jsonl_to_phase_csv \
    --jsonl "$VLM_JSONL" \
    --out-csv "$OUT_CSV" \

  echo "✅ Final: $OUT_CSV"
  CASE_END_TIME=$(date +%s)
  CASE_ELAPSED=$((CASE_END_TIME - CASE_START_TIME))

  echo "----------------------------------------"
  echo "✅ Final output: $OUT_CSV"
  echo "⏱️  CASE TIME: ${CASE_ELAPSED}s ($((${CASE_ELAPSED}/60))m $((${CASE_ELAPSED}%60))s)"
  echo "----------------------------------------"
done

echo "🎉 DONE"

SCRIPT_END_TIME=$(date +%s)
SCRIPT_ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

echo
echo "========================================"
echo "🎉 PIPELINE FINISHED"
echo "⏱️  TOTAL TIME: ${SCRIPT_ELAPSED}s ($((${SCRIPT_ELAPSED}/60))m $((${SCRIPT_ELAPSED}%60))s)"
echo "========================================"
