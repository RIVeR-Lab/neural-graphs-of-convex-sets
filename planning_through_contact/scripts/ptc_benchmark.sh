#!/usr/bin/env bash
# Run planar pushing benchmarks and generate results-table PDFs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
OUTPUT_DIR="planning_through_contact/results/benchmark"
EXTRA_ARGS=()

usage() {
    echo "Usage: bash planning_through_contact/scripts/ptc_benchmark.sh [--box] [--tee] [--output-dir PATH] [extra benchmark args...]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

benchmark_body() {
    local body="$1"

    echo ""
    echo "=== Planar Pushing Benchmark ($body) ==="
    echo "  Output dir : $OUTPUT_DIR"
    echo ""

    python -m planning_through_contact.scripts.planar_pushing.benchmark_planar_pushing \
        --body "$body" \
        --output_dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}"

    local latest_dir
    latest_dir="$(ls -td "$OUTPUT_DIR"/"${body}"_* 2>/dev/null | head -1 || true)"
    if [[ -z "$latest_dir" ]]; then
        echo "[error] No benchmark output found for $body in $OUTPUT_DIR." >&2
        return 1
    fi

    local summary_json="$latest_dir/summary.json"
    echo ""
    echo "Generating results table PDF: $summary_json"
    python -m planning_through_contact.scripts.planar_pushing.generate_results_table \
        "$summary_json" \
        --title "Planar Pushing Benchmark - $body"
    echo "Results table: $latest_dir/results_table.pdf"
}

$RUN_BOX && benchmark_body "sugar_box"
$RUN_TEE && benchmark_body "tee"
