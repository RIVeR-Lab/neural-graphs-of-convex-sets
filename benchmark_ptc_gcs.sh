#!/usr/bin/env bash
# Run planning-through-contact planar pushing benchmarks for box and tee back-to-back.

set -euo pipefail

MAX_INSTANCES=""
DEVICE="cpu"
OUTPUT_DIR="planning_through_contact/results/benchmark"
LAST_PDF=""

usage() {
    cat <<'EOF'
Usage: bash benchmark_ptc_gcs.sh [options]

Options:
  --max_instances N   Cap number of test plans per body (maps to --max_test_plans)
  --device DEVICE     cpu, cuda, or auto (default: cpu)
  --output_dir DIR    Output directory (default: planning_through_contact/results/benchmark)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --max_instances|--max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --output_dir|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

cd "$(dirname "$0")"

common_args=(
    --device "$DEVICE"
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_INSTANCES" ]]; then
    common_args+=(--max_test_plans "$MAX_INSTANCES")
fi

benchmark_body() {
    local body="$1"
    local label="$2"

    echo "=== Planar pushing benchmark ($label) ==="
    python -m planning_through_contact.scripts.planar_pushing.benchmark_planar_pushing \
        --body "$body" \
        "${common_args[@]}"

    local latest_dir
    latest_dir="$(ls -td "$OUTPUT_DIR"/"${body}"_* 2>/dev/null | head -1 || true)"
    if [[ -z "$latest_dir" ]]; then
        echo "[error] No benchmark output found for $body in $OUTPUT_DIR." >&2
        exit 1
    fi

    echo ""
    echo "Generating results table PDF: $latest_dir/summary.json"
    python -m planning_through_contact.scripts.planar_pushing.generate_results_table \
        "$latest_dir/summary.json" \
        --title "Planar Pushing Benchmark - $label"

    LAST_PDF="$(pwd)/$latest_dir/results_table.pdf"
}

echo "=== Planning through contact — sugar_box ==="
benchmark_body sugar_box box
BOX_PDF="$LAST_PDF"

echo ""
echo "=== Planning through contact — tee ==="
benchmark_body tee tee
TEE_PDF="$LAST_PDF"

echo ""
echo "Wrote:"
echo "  $BOX_PDF"
echo "  $TEE_PDF"
