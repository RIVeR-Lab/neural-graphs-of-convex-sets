#!/usr/bin/env bash
# Run GNN training on the dataset in planning_through_contact/dataset/data/ (e.g. 1 train + 1 test).
# Run from repo root: ./planning_through_contact/scripts/planar_pushing/train.sh [extra args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data}"
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

echo "Training: h5=$H5_PATH node_features=$NODE_FEATURES_CSV"
exec python3 planning_through_contact/scripts/train_gcs_gnn.py \
  --h5_path "$H5_PATH" \
  --node_features_csv "$NODE_FEATURES_CSV" \
  "$@"
