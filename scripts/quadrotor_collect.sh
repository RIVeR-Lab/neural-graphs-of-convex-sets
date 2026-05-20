#!/usr/bin/env bash
# Collect the quadrotor GCS training dataset.
#
# Usage:
#   bash scripts/quadrotor_collect.sh [options]
#
# Options:
#   --output_dir DIR    Where to write the HDF5 (default: quadrotor/dataset)
#   --splits SPLITS     Comma-separated splits to generate (default: train,val,test)
#   --seed INT          Base RNG seed (default: 0)
#
# Example:
#   bash scripts/quadrotor_collect.sh
#   bash scripts/quadrotor_collect.sh --splits train --output_dir /tmp/quadrotor_dataset

set -euo pipefail

OUTPUT_DIR="quadrotor/dataset"
SPLITS=""
SEED=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --splits)     SPLITS="$2";     shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXTRA_ARGS=""
if [[ -n "$SPLITS" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --splits $SPLITS"
fi

echo "=== Quadrotor Data Collection ==="
echo "  Output dir : $OUTPUT_DIR"
echo "  Seed       : $SEED"
echo ""

python scripts/collect_quadrotor_data.py \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    $EXTRA_ARGS