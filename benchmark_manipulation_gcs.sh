#!/usr/bin/env bash
# Run manipulation convex and nonconvex GCS test-set benchmarks back-to-back.

set -euo pipefail

MAX_INSTANCES=""
SPLIT="test"
DEVICE="cpu"
OUTPUT_DIR="manipulation/results/shelf_viz/dataset_benchmark"

usage() {
    cat <<'EOF'
Usage: bash benchmark_manipulation_gcs.sh [options]

Options:
  --max_instances N   Number of reported test instances per planner
  --split SPLIT       Dataset split: train, val, or test (default: test)
  --device DEVICE     cpu or cuda (default: cpu)
  --output_dir DIR    Output directory (default: manipulation/results/shelf_viz/dataset_benchmark)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --max_instances|--max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --output_dir|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

cd "$(dirname "$0")"

common_args=(
    --split "$SPLIT"
    --device "$DEVICE"
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_INSTANCES" ]]; then
    common_args+=(--max_instances "$MAX_INSTANCES")
fi

echo "=== Manipulation convex GCS benchmark ==="
python manipulation/scripts/benchmark_manipulation_dataset.py \
    --planner convex \
    "${common_args[@]}"

echo ""
echo "=== Manipulation nonconvex GCS benchmark ==="
python manipulation/scripts/benchmark_manipulation_dataset.py \
    --planner nonconvex \
    "${common_args[@]}"

echo ""
echo "Wrote:"
echo "  $(pwd)/$OUTPUT_DIR/results_table_convex.pdf"
echo "  $(pwd)/$OUTPUT_DIR/results_table_nonconvex.pdf"
