#!/usr/bin/env bash
# Train PointNet flow GNN (Phase 1) + RankNet (Phase 2) for quadrotor GCS.
#
# Usage:
#   bash quadrotor/scripts/quadrotor_train.sh --planner convex
#   bash quadrotor/scripts/quadrotor_train.sh --planner nonconvex --no_wandb
#   bash quadrotor/scripts/quadrotor_train.sh convex --ckpt_dir quadrotor/checkpoints/quadrotor_convex
#
# Datasets (auto-selected from --planner):
#   convex    -> quadrotor/dataset/quadrotor_gcs_convex.h5
#   nonconvex -> quadrotor/dataset/quadrotor_gcs_nonconvex.h5
#
# Checkpoints (default):
#   quadrotor/checkpoints/quadrotor_convex/    or quadrotor/checkpoints/quadrotor_nonconvex/
#
# Extra flags are forwarded to scripts/train_gcs_flow.py (e.g. --no_wandb, --h5_path).

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
        convex|nonconvex) PLANNER="$1"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$PLANNER" != "convex" && "$PLANNER" != "nonconvex" ]]; then
    echo "Unknown planner: $PLANNER (use convex or nonconvex)" >&2
    exit 1
fi

if [[ -z "$CKPT_DIR" ]]; then
    CKPT_DIR="quadrotor/checkpoints/quadrotor_${PLANNER}"
fi

case "$PLANNER" in
    convex)    H5_PATH="quadrotor/dataset/quadrotor_gcs_convex.h5" ;;
    nonconvex) H5_PATH="quadrotor/dataset/quadrotor_gcs_nonconvex.h5" ;;
esac

echo "=== Quadrotor Training ($PLANNER) ==="
echo "  Dataset     : $H5_PATH"
echo "  Checkpoints : $CKPT_DIR"
echo ""

"$PY" scripts/train_gcs_flow.py \
    --problem quadrotor \
    --planner "$PLANNER" \
    --h5_path "$H5_PATH" \
    --phase both \
    --ckpt_dir "$CKPT_DIR" \
    --wandb_project gnn-for-gcs-quadrotor \
    "${EXTRA_ARGS[@]}"
