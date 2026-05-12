#!/usr/bin/env bash
# Train Flow GNN (Phase 1) then RankNet (Phase 2) for quadrotor motion planning.
#
# Usage:
#   bash scripts/quadrotor_train.sh [options]
#
# Options:
#   --h5_path PATH          HDF5 dataset path (default: quadrotor/dataset/quadrotor_gcs_dataset.h5)
#   --ckpt_dir DIR          Checkpoint root (default: quadrotor/checkpoints)
#   --wandb_entity ENTITY   W&B entity (default: unset)
#   --wandb_project PROJECT W&B project (default: gnn-for-gcs-quadrotor)
#   --max_epochs_gnn INT    Max epochs for Phase 1 (default: 500)
#   --max_epochs_ranknet INT Max epochs for Phase 2 (default: 100)
#   --no_wandb              Use TensorBoard instead of W&B
#
# Example:
#   bash scripts/quadrotor_train.sh
#   bash scripts/quadrotor_train.sh --wandb_entity my-team --no_wandb

set -euo pipefail

H5_PATH="quadrotor/dataset/quadrotor_gcs_dataset.h5"
CKPT_DIR="quadrotor/checkpoints"
WANDB_ENTITY=""
WANDB_PROJECT="gnn-for-gcs-quadrotor"
MAX_EPOCHS_GNN=500
MAX_EPOCHS_RANKNET=100
NO_WANDB=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --h5_path)            H5_PATH="$2";            shift 2 ;;
        --ckpt_dir)           CKPT_DIR="$2";           shift 2 ;;
        --wandb_entity)       WANDB_ENTITY="$2";       shift 2 ;;
        --wandb_project)      WANDB_PROJECT="$2";      shift 2 ;;
        --max_epochs_gnn)     MAX_EPOCHS_GNN="$2";     shift 2 ;;
        --max_epochs_ranknet) MAX_EPOCHS_RANKNET="$2"; shift 2 ;;
        --no_wandb)           NO_WANDB="--no_wandb";   shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

FLOW_CKPT="$CKPT_DIR/quadrotor_gnn/quadrotor_flow_gnn.ckpt"

WANDB_ARGS=""
if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS="--wandb_entity $WANDB_ENTITY"
fi

echo "=== Quadrotor Training ==="
echo "  Dataset    : $H5_PATH"
echo "  Checkpoints: $CKPT_DIR"
echo "  W&B project: $WANDB_PROJECT"
echo ""

# ── Phase 1: Flow GNN ────────────────────────────────────────────────────────
echo ">>> Phase 1: Flow GNN (max_epochs=$MAX_EPOCHS_GNN)"
python scripts/train_quadrotor_gnn.py \
    --h5_path "$H5_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --max_epochs "$MAX_EPOCHS_GNN" \
    --wandb_run_name "quadrotor_gnn" \
    $WANDB_ARGS \
    $NO_WANDB

echo ""
echo ">>> Phase 1 complete. Checkpoint: $FLOW_CKPT"
echo ""

# ── Phase 2: RankNet ─────────────────────────────────────────────────────────
echo ">>> Phase 2: RankNet (max_epochs=$MAX_EPOCHS_RANKNET)"
python scripts/train_quadrotor_ranknet.py \
    --h5_path "$H5_PATH" \
    --flow_ckpt "$FLOW_CKPT" \
    --ckpt_dir "$CKPT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --max_epochs "$MAX_EPOCHS_RANKNET" \
    --wandb_run_name "quadrotor_ranknet" \
    $WANDB_ARGS \
    $NO_WANDB

echo ""
echo ">>> Phase 2 complete."
echo ">>> All checkpoints saved under $CKPT_DIR"