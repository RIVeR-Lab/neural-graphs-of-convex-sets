#!/usr/bin/env bash
# Run GNN inference (planning) using a trained checkpoint. Tests the full pipeline.
# Run from repo root: ./planning_through_contact/scripts/planar_pushing/test.sh [extra args...]
# Default checkpoint: checkpoints/gcs_gnn/last.ckpt (override with CKPT_PATH=...).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data}"
NODE_FEATURES_CSV="${NODE_FEATURES_CSV:-$DATA_DIR/node_features.csv}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints/gcs_gnn/last.ckpt}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Error: Checkpoint not found: $CKPT_PATH" >&2
  echo "Train first: ./planning_through_contact/scripts/planar_pushing/train.sh --max_epochs 2" >&2
  exit 1
fi
if [[ ! -f "$NODE_FEATURES_CSV" ]]; then
  echo "Error: node_features.csv not found: $NODE_FEATURES_CSV" >&2
  exit 1
fi

echo "Inference: ckpt=$CKPT_PATH node_features=$NODE_FEATURES_CSV"
exec python3 planning_through_contact/scripts/planar_pushing/plan_with_gnn.py \
  --ckpt_path "$CKPT_PATH" \
  --node_features_csv "$NODE_FEATURES_CSV" \
  --num 1 \
  "$@"
