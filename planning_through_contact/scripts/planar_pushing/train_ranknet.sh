#!/usr/bin/env bash
set -euo pipefail

# Train the RankNet path ranker for one planar pushing body.
# Requires a trained flow checkpoint at checkpoints/gcs_gnn/${BODY}_flow.ckpt unless FLOW_CKPT_PATH is set.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

BODY="${BODY:-sugar_box}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data/$BODY}"
FLOW_CKPT_PATH="${FLOW_CKPT_PATH:-$REPO_ROOT/checkpoints/gcs_gnn/${BODY}_flow.ckpt}"

if [[ ! -f "$DATA_DIR/gcs_solutions.h5" ]]; then
  echo "Error: H5 not found: $DATA_DIR/gcs_solutions.h5" >&2
  exit 1
fi
if [[ ! -f "$DATA_DIR/node_features.csv" ]]; then
  echo "Error: node_features.csv not found: $DATA_DIR/node_features.csv" >&2
  exit 1
fi
if [[ ! -f "$FLOW_CKPT_PATH" ]]; then
  echo "Error: flow checkpoint not found: $FLOW_CKPT_PATH" >&2
  exit 1
fi

echo "Training RankNet: body=$BODY data=$DATA_DIR flow_ckpt=$FLOW_CKPT_PATH"
exec python3 planning_through_contact/scripts/train_ranknet.py \
  --body "$BODY" \
  --flow_ckpt_path "$FLOW_CKPT_PATH" \
  --wandb_run_name "${BODY}_ranknet" \
  "$@"
