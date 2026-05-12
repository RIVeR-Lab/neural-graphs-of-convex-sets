#!/usr/bin/env bash
# Run flow-GNN training for one planar pushing body, regressing on SDP (phi_star).
# Run from repo root: ./planning_through_contact/scripts/planar_pushing/train.sh [extra args...]
# Defaults: BODY=sugar_box, W&B on, 500 epochs, batch 64, no early stopping.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

BODY="${BODY:-sugar_box}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data/$BODY}"
H5_PATH="${H5_PATH:-$DATA_DIR/gcs_solutions.h5}"
NODE_FEATURES_CSV="${NODE_FEATURES_CSV:-$DATA_DIR/node_features.csv}"

if [[ ! -f "$H5_PATH" ]]; then
  echo "Error: H5 not found: $H5_PATH" >&2
  echo "Run planning_through_contact/dataset/run_dataset_generation.sh first." >&2
  exit 1
fi
if [[ ! -f "$NODE_FEATURES_CSV" ]]; then
  echo "Error: node_features.csv not found: $NODE_FEATURES_CSV" >&2
  exit 1
fi

echo "Training flow GNN: body=$BODY h5=$H5_PATH node_features=$NODE_FEATURES_CSV"
exec python3 planning_through_contact/scripts/train_gcs_gnn.py \
  --body "$BODY" \
  --h5_path "$H5_PATH" \
  --node_features_csv "$NODE_FEATURES_CSV" \
  --target sdp \
  --wandb_run_name "${BODY}_flow_gnn" \
  --max_epochs 500 \
  --batch_size 64 \
  --no_early_stopping \
  "$@"
