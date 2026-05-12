#!/usr/bin/env bash
# Generate training-ready quadrotor GCS datasets (convex and/or nonconvex HDF5).
#
# Usage:
#   bash quadrotor/scripts/quadrotor_collect.sh
#   bash quadrotor/scripts/quadrotor_collect.sh --planners convex
#   bash quadrotor/scripts/quadrotor_collect.sh nonconvex --train-count 10
#
# Writes HDF5 under quadrotor/dataset/ with graph solutions, candidate paths,
# and per-region halfspaces (A, b) for PointNet training.
#
# Extra flags are forwarded to generate_quadrotor_dataset.py (e.g. --seed, --overwrite).

set -euo pipefail

cd "$(dirname "$0")/../.."

PY=${PY:-python3}
PLANNERS="both"
OUTPUT_DIR="quadrotor/dataset"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --planners) PLANNERS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        convex|nonconvex|both) PLANNERS="$1"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Quadrotor Data Collection ==="
echo "  Planners   : $PLANNERS"
echo "  Output dir : $OUTPUT_DIR"
echo ""

exec "$PY" quadrotor/scripts/generate_quadrotor_dataset.py \
    --planners "$PLANNERS" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
