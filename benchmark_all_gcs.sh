#!/usr/bin/env bash
# Run quadrotor, manipulation, and planning-through-contact benchmarks in sequence.

set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: bash benchmark_all_gcs.sh [options]

Runs, in order:
  benchmark_quadrotor_gcs.sh
  benchmark_manipulation_gcs.sh
  benchmark_ptc_gcs.sh

Options (forwarded to all three):
  --max_instances N   Cap instances / test plans per benchmark
  --device DEVICE     cpu or cuda (ptc also accepts auto)
  --help              Show this help

Per-benchmark output dirs are unchanged; see each script for defaults.
EOF
}

MAX_INSTANCES=""
DEVICE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --max_instances|--max-instances)
            MAX_INSTANCES="$2"
            EXTRA_ARGS+=(--max_instances "$2")
            shift 2
            ;;
        --device)
            DEVICE="$2"
            EXTRA_ARGS+=(--device "$2")
            shift 2
            ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

run() {
    local script="$1"
    echo ""
    echo "################################################################"
    echo "# $script"
    echo "################################################################"
    bash "$script" "${EXTRA_ARGS[@]}"
}

run benchmark_quadrotor_gcs.sh
run benchmark_manipulation_gcs.sh
run benchmark_ptc_gcs.sh

echo ""
echo "All benchmarks complete."
