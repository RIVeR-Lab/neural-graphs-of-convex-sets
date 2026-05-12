#!/usr/bin/env bash
# Run the manipulation circle benchmark with manipulation-local checkpoints.

set -euo pipefail

cd "$(dirname "$0")/../.."

PY=${PY:-python3}
PLANNER="convex"
DEVICE="cuda"
OUTPUT_DIR="manipulation/results/shelf_viz/circle_demo"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --planner) PLANNER="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        convex|nonconvex) PLANNER="$1"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$PLANNER" != "convex" && "$PLANNER" != "nonconvex" ]]; then
    echo "Unknown planner: $PLANNER (use convex or nonconvex)" >&2
    exit 1
fi

CKPT_DIR="manipulation/checkpoints/manipulation_${PLANNER}"

echo "=== Manipulation Benchmark ($PLANNER) ==="
echo "  Checkpoints : $CKPT_DIR"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Device      : $DEVICE"
echo ""

exec "$PY" scripts/benchmark_manipulation_circle_warm.py \
    --planner "$PLANNER" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    --flow-ckpt "$CKPT_DIR/manipulation_${PLANNER}_flow_gnn.ckpt" \
    --ranknet-ckpt "$CKPT_DIR/manipulation_${PLANNER}_ranknet.ckpt" \
    "${EXTRA_ARGS[@]}"
