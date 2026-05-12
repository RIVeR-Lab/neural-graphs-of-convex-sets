#!/usr/bin/env bash
# Train PointNet flow GNN and RankNet for manipulation GCS.

set -euo pipefail

cd "$(dirname "$0")/../.."

PY=${PY:-python3}
PLANNER="convex"
CKPT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --planner) PLANNER="$2"; shift 2 ;;
        --ckpt_dir) CKPT_DIR="$2"; shift 2 ;;
        --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
        convex|nonconvex) PLANNER="$1"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$PLANNER" != "convex" && "$PLANNER" != "nonconvex" ]]; then
    echo "Unknown planner: $PLANNER (use convex or nonconvex)" >&2
    exit 1
fi

if [[ -z "$CKPT_DIR" ]]; then
    CKPT_DIR="manipulation/checkpoints/manipulation_${PLANNER}"
fi

case "$PLANNER" in
    convex)    H5_PATH="manipulation/dataset/manipulation_gcs_convex.h5" ;;
    nonconvex) H5_PATH="manipulation/dataset/manipulation_gcs_nonconvex.h5" ;;
esac

echo "=== Manipulation Training ($PLANNER) ==="
echo "  Dataset     : $H5_PATH"
echo "  Checkpoints : $CKPT_DIR"
echo ""

"$PY" scripts/train_gcs_flow.py \
    --problem manipulation \
    --planner "$PLANNER" \
    --h5_path "$H5_PATH" \
    --phase both \
    --ckpt_dir "$CKPT_DIR" \
    --wandb_project gnn-for-gcs-manipulation \
    "${EXTRA_ARGS[@]}"
