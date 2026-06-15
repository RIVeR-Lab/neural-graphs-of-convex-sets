#!/usr/bin/env bash
# Collect quadrotor GCS generalization dataset (convex + nonlinear HDF5 files).
#
# Usage:
#   bash scripts/quadrotor_collect_generalization.sh [options]
#
# Options override the defaults below (same flags as collect_quadrotor_generalization.py).
#
# Example:
#   bash scripts/quadrotor_collect_generalization.sh
#   bash scripts/quadrotor_collect_generalization.sh --splits train --train-count 10

set -euo pipefail

# ---------- dataset layout ----------
OUTPUT_DIR="quadrotor/dataset"
SEED=0
SPLITS="train,val,test"

# train: 4x4 buildings
TRAIN_SIZE=4
TRAIN_COUNT=500

# val: 3x3 and 5x5 (OOD scale)
VAL_3_COUNT=50
VAL_5_COUNT=50

# test: 3x3 and 5x5 (OOD scale)
TEST_3_COUNT=50
TEST_5_COUNT=50

# max start/goal sampling attempts per building before moving on
MAX_BUILDING_ATTEMPTS=5

# ---------- parse CLI overrides ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)           OUTPUT_DIR="$2";           shift 2 ;;
        --seed)                 SEED="$2";                 shift 2 ;;
        --splits)               SPLITS="$2";               shift 2 ;;
        --train-size)           TRAIN_SIZE="$2";           shift 2 ;;
        --train-count)          TRAIN_COUNT="$2";          shift 2 ;;
        --val-3-count)          VAL_3_COUNT="$2";          shift 2 ;;
        --val-5-count)          VAL_5_COUNT="$2";          shift 2 ;;
        --test-3-count)         TEST_3_COUNT="$2";         shift 2 ;;
        --test-5-count)         TEST_5_COUNT="$2";         shift 2 ;;
        --max-building-attempts) MAX_BUILDING_ATTEMPTS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== Quadrotor Generalization Data Collection ==="
echo "  Output dir            : $OUTPUT_DIR"
echo "  Seed                  : $SEED"
echo "  Splits                : $SPLITS"
echo "  Train                 : ${TRAIN_COUNT} x ${TRAIN_SIZE}x${TRAIN_SIZE}"
echo "  Val                   : ${VAL_3_COUNT} x 3x3, ${VAL_5_COUNT} x 5x5"
echo "  Test                  : ${TEST_3_COUNT} x 3x3, ${TEST_5_COUNT} x 5x5"
echo "  Max attempts/building : $MAX_BUILDING_ATTEMPTS"
echo ""

python scripts/collect_quadrotor_generalization.py \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --splits "$SPLITS" \
    --train-size "$TRAIN_SIZE" \
    --train-count "$TRAIN_COUNT" \
    --val-3-count "$VAL_3_COUNT" \
    --val-5-count "$VAL_5_COUNT" \
    --test-3-count "$TEST_3_COUNT" \
    --test-5-count "$TEST_5_COUNT" \
    --max-building-attempts "$MAX_BUILDING_ATTEMPTS"
