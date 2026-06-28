#!/usr/bin/env bash
# Collect IIWA shelf GCS dataset (linear + nonlinear HDF5 files).
#
# Usage:
#   bash scripts/iiwa_collect_data.sh
#   bash scripts/iiwa_collect_data.sh --splits train --train-count 5

set -euo pipefail

OUTPUT_DIR="manipulation/data"
SEED=0
SPLITS="train,val,test"
TRAIN_COUNT=500
VAL_COUNT=100
TEST_COUNT=100
PLANNER="both"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --splits)       SPLITS="$2";       shift 2 ;;
        --train-count)  TRAIN_COUNT="$2";  shift 2 ;;
        --val-count)    VAL_COUNT="$2";    shift 2 ;;
        --test-count)   TEST_COUNT="$2";   shift 2 ;;
        --planner)      PLANNER="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== IIWA Shelf GCS Data Collection ==="
echo "  Output dir : $OUTPUT_DIR"
echo "  Seed       : $SEED"
echo "  Splits     : $SPLITS"
echo "  Counts     : train=$TRAIN_COUNT val=$VAL_COUNT test=$TEST_COUNT"
echo "  Planner    : $PLANNER"
echo ""

python scripts/collect_iiwa_shelf_data.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --splits "$SPLITS" \
    --train-count "$TRAIN_COUNT" \
    --val-count "$VAL_COUNT" \
    --test-count "$TEST_COUNT" \
    --planner "$PLANNER"
