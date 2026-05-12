#!/usr/bin/env bash
# Run quadrotor convex and nonconvex GCS benchmarks back-to-back.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ananya/Documents/neural-graphs-of-convex-sets}"
MAX_INSTANCES=""
SPLIT="test"
DEVICE="cpu"
OUTPUT_DIR="quadrotor/results"

usage() {
    cat <<'EOF'
Usage: bash ~/benchmark_quadrotor_gcs.sh [options]

Options:
  --max_instances N   Number of reported test instances per planner
  --split SPLIT       Dataset split: train, val, or test (default: test)
  --device DEVICE     cpu or cuda (default: cpu)
  --output_dir DIR    Output directory (default: quadrotor/results)
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

cd "$REPO_ROOT"

common_args=(
    --split "$SPLIT"
    --device "$DEVICE"
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_INSTANCES" ]]; then
    common_args+=(--max_instances "$MAX_INSTANCES")
fi

echo "=== Quadrotor convex GCS benchmark ==="
bash quadrotor/scripts/quadrotor_benchmark.sh --planner convex "${common_args[@]}"

echo ""
echo "=== Quadrotor nonconvex GCS benchmark ==="
bash quadrotor/scripts/quadrotor_benchmark.sh --planner nonconvex "${common_args[@]}"

echo ""
echo "Wrote:"
echo "  $REPO_ROOT/$OUTPUT_DIR/results_table_convex.pdf"
echo "  $REPO_ROOT/$OUTPUT_DIR/results_table_nonconvex.pdf"
