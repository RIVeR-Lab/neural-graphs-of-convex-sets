#!/usr/bin/env bash
# Generate training-ready manipulation GCS datasets.

set -euo pipefail

cd "$(dirname "$0")/../.."

PY=${PY:-python3}
PLANNERS="both"
OUTPUT_DIR="manipulation/dataset"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --planners) PLANNERS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        convex|nonconvex|both) PLANNERS="$1"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Manipulation Data Collection ==="
echo "  Planners   : $PLANNERS"
echo "  Output dir : $OUTPUT_DIR"
echo ""

exec "$PY" manipulation/scripts/generate_manipulation_dataset.py \
    --planners "$PLANNERS" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
