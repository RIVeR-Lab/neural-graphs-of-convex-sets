#!/usr/bin/env bash
# Run the full planar pushing benchmark and generate the results PDF.
#
# Usage:
#   bash scripts/run_full_benchmark.sh [--box] [--tee] [--skip_trajopt] [extra benchmark args...]
#
# --box   Run benchmark for the sugar_box body (run first if both specified)
# --tee   Run benchmark for the tee body
#
# At least one of --box or --tee must be provided.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    echo "Usage: bash scripts/run_full_benchmark.sh [--box] [--tee] [extra args...]"
    echo "At least one of --box or --tee is required."
    exit 1
fi

run_benchmark() {
    local BODY="$1"
    echo ""
    echo "========================================"
    echo " Planar Pushing Benchmark — $BODY"
    echo "========================================"
    echo ""

    python -m planning_through_contact.scripts.planar_pushing.benchmark_planar_pushing \
        --body "$BODY" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

    LATEST_DIR=$(ls -td benchmark_results/"$BODY"_* 2>/dev/null | head -1)
    if [[ -z "$LATEST_DIR" ]]; then
        echo "[error] No benchmark output found for $BODY."
        return 1
    fi

    SUMMARY_JSON="$LATEST_DIR/summary.json"
    echo ""
    echo "Generating results table PDF — $SUMMARY_JSON"

    python -m planning_through_contact.scripts.planar_pushing.generate_results_table \
        "$SUMMARY_JSON" \
        --title "Planar Pushing Benchmark — $BODY"

    echo "Results table: $LATEST_DIR/results_table.pdf"
}

$RUN_BOX && run_benchmark "sugar_box"
$RUN_TEE && run_benchmark "tee"