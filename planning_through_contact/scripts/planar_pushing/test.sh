#!/usr/bin/env bash
# Run GNN inference (planning) using a trained checkpoint. Tests the full pipeline.
# Run from repo root: ./planning_through_contact/scripts/planar_pushing/test.sh [extra args...]
# Checkpoint: defaults to checkpoints/gcs_gnn/${BODY}_flow.ckpt.
# Override with CKPT_PATH=...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

BODY="${BODY:-sugar_box}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data/$BODY}"
NODE_FEATURES_CSV="${NODE_FEATURES_CSV:-$DATA_DIR/node_features.csv}"
PLAN_INDEX_CSV="${PLAN_INDEX_CSV:-$DATA_DIR/global_features.csv}"
H5_PATH="${H5_PATH:-$DATA_DIR/gcs_solutions.h5}"

CKPT_DIR="$REPO_ROOT/checkpoints/gcs_gnn"
if [[ ! -v CKPT_PATH ]]; then
  CKPT_PATH="$CKPT_DIR/${BODY}_flow.ckpt"
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Error: Checkpoint not found: $CKPT_PATH" >&2
  echo "Download: ./planning_through_contact/dataset/download_box_pushing_assets.sh" >&2
  echo "Or train: ./planning_through_contact/scripts/planar_pushing/train.sh" >&2
  exit 1
fi
if [[ ! -f "$NODE_FEATURES_CSV" ]]; then
  echo "Error: node_features.csv not found: $NODE_FEATURES_CSV" >&2
  exit 1
fi
if [[ ! -f "$PLAN_INDEX_CSV" ]]; then
  echo "Error: plan index (global_features.csv) not found: $PLAN_INDEX_CSV" >&2
  echo "Required for --use_test_plans. Generate data with run_dataset_generation.sh." >&2
  exit 1
fi

# Run on test split from plan index (evaluate on held-out labels; test loss + ground-truth SVG when H5 present)
# Rounding: QP = project predicted flows to flow polytope before path sampling.
# Videos (prediction.mp4, ground_truth.mp4) are saved by default; use NO_SAVE_VIDEO=1 or --no_save_video to disable.
VIDEO_ARGS=()
[[ -n "${NO_SAVE_VIDEO:-}" && "${NO_SAVE_VIDEO}" != "0" ]] && VIDEO_ARGS=(--no_save_video)
echo "Inference (test split): body=$BODY ckpt=$CKPT_PATH plan_index=$PLAN_INDEX_CSV h5=$H5_PATH rounding_flow=qp"
exec python3 planning_through_contact/scripts/planar_pushing/plan_with_gnn.py \
  --ckpt_path "$CKPT_PATH" \
  --node_features_csv "$NODE_FEATURES_CSV" \
  --use_test_plans \
  --plan_index_csv "$PLAN_INDEX_CSV" \
  --h5_path "$H5_PATH" \
  --rounding_flow qp \
  "${VIDEO_ARGS[@]}" \
  "$@"
