#!/usr/bin/env bash
set -euo pipefail

# Generate dataset: plan index -> static node features -> collect solutions into HDF5.
# Run from repo root.
#
# Choose object with BODY=sugar_box or BODY=tee (default: sugar_box).
# Override split sizes with NUM_TRAIN, NUM_VAL, NUM_TEST (defaults: 1 / 0 / 1 for a tiny smoke run).
# Match `create_plan_index.py` full defaults with: NUM_TRAIN=500 NUM_VAL=100 NUM_TEST=100
# Override candidate paths with NUM_CANDIDATE_PATHS (default: 100).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BODY="${BODY:-sugar_box}"
case "$BODY" in
  sugar_box|tee) ;;
  *)
    echo "Error: BODY must be 'sugar_box' or 'tee' (got '$BODY')." >&2
    exit 1
    ;;
esac
NUM_TRAIN="${NUM_TRAIN:-1}"
NUM_VAL="${NUM_VAL:-0}"
NUM_TEST="${NUM_TEST:-1}"
NUM_CANDIDATE_PATHS="${NUM_CANDIDATE_PATHS:-100}"
NUM_PLANS=$((NUM_TRAIN + NUM_VAL + NUM_TEST))

DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data/$BODY}"
PLAN_INDEX="$DATA_DIR/global_features.csv"
H5_PATH="${H5_PATH:-$DATA_DIR/gcs_solutions.h5}"

echo "Dataset body: $BODY"
echo "Data dir: $DATA_DIR"
echo "1. Plan index (train=$NUM_TRAIN val=$NUM_VAL test=$NUM_TEST -> total=$NUM_PLANS)..."
python3 planning_through_contact/dataset/create_plan_index.py \
  --num_train "$NUM_TRAIN" \
  --num_val "$NUM_VAL" \
  --num_test "$NUM_TEST" \
  --body "$BODY" \
  --output_dir "$DATA_DIR" \
  --output_stem global_features

echo "2. Static node features..."
python3 planning_through_contact/dataset/create_static_node_features.py \
  --body "$BODY" \
  --output_path "$DATA_DIR/node_features.csv"

echo "3. Collect solutions one plan per Python process (plan_id 0..$((NUM_PLANS - 1)))..."
for ((PLAN_ID=0; PLAN_ID<NUM_PLANS; PLAN_ID++)); do
  NEXT_PLAN_ID=$((PLAN_ID + 1))
  echo "  Collecting plan_id=$PLAN_ID..."
  python3 planning_through_contact/dataset/collect_solutions.py \
    --plan_index_csv "$PLAN_INDEX" \
    --start_id "$PLAN_ID" \
    --end_id "$NEXT_PLAN_ID" \
    --h5_path "$H5_PATH" \
    --num_candidate_paths "$NUM_CANDIDATE_PATHS"
done

echo "Done. H5: $H5_PATH"
