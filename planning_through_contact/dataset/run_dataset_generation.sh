#!/usr/bin/env bash
# Generate dataset: 2 plans (1 train, 1 test) -> static node features -> collect solutions into HDF5.
# Run from repo root. Validation and testing both use the test split.



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

NUM="${NUM:-2}"
TEST_FRAC="${TEST_FRAC:-0.5}"
BODY="${BODY:-sugar_box}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data}"
PLAN_INDEX="$DATA_DIR/global_features.csv"
H5_PATH="${H5_PATH:-$DATA_DIR/gcs_solutions.h5}"

echo "1. Plan index (num=$NUM, test_frac=$TEST_FRAC -> 1 train, 1 test)..."
python3 planning_through_contact/dataset/create_plan_index.py \
  --num "$NUM" \
  --test_frac "$TEST_FRAC" \
  --body "$BODY" \
  --output_dir "$DATA_DIR" \
  --output_stem global_features

echo "2. Static node features..."
python3 planning_through_contact/dataset/create_static_node_features.py \
  --body "$BODY" \
  --output_path "$DATA_DIR/node_features.csv"

echo "3. Collect solutions (plan_id 0..$NUM)..."
python3 planning_through_contact/dataset/collect_solutions.py \
  --plan_index_csv "$PLAN_INDEX" \
  --start_id 0 \
  --end_id "$NUM" \
  --h5_path "$H5_PATH"

echo "Done. H5: $H5_PATH (1 train, 1 test sample)."
