#!/usr/bin/env bash
# Run 4 test-set instances with contact-aware video visualization at 1920x1080.
# Usage (from repo root):
#   ./planning_through_contact/scripts/planar_pushing/test_4_instances_record_videos.sh [extra args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data/box_pushing}"
NODE_FEATURES_CSV="${NODE_FEATURES_CSV:-$DATA_DIR/node_features.csv}"
PLAN_INDEX_CSV="${PLAN_INDEX_CSV:-$DATA_DIR/global_features.csv}"
H5_PATH="${H5_PATH:-$DATA_DIR/gcs_solutions.h5}"

CKPT_DIR="$REPO_ROOT/checkpoints/gcs_gnn"
CKPT_PATH="${CKPT_PATH:-$CKPT_DIR/box_pushing.ckpt}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Error: Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi
if [[ ! -f "$NODE_FEATURES_CSV" ]]; then
  echo "Error: node_features.csv not found: $NODE_FEATURES_CSV" >&2
  exit 1
fi
if [[ ! -f "$PLAN_INDEX_CSV" ]]; then
  echo "Error: plan index CSV not found: $PLAN_INDEX_CSV" >&2
  exit 1
fi

echo "Running 4 test instances with contact legend video output."
exec python3 planning_through_contact/scripts/planar_pushing/plan_with_gnn.py \
  --ckpt_path "$CKPT_PATH" \
  --node_features_csv "$NODE_FEATURES_CSV" \
  --use_test_plans \
  --max_test_plans 4 \
  --plan_index_csv "$PLAN_INDEX_CSV" \
  --h5_path "$H5_PATH" \
  --rounding_flow qp \
  --show_contact_legend \
  --video_width_px 1920 \
  --video_height_px 1080 \
  "$@"
