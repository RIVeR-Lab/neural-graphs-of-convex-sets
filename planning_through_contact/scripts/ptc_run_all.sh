#!/usr/bin/env bash
# Run planar pushing benchmark tables, figures, and videos.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
DO_TABLE=true
DO_PDF=true
DO_VIDEO=true
SKIP_TRAJOPT=()

usage() {
    echo "Usage: bash planning_through_contact/scripts/ptc_run_all.sh [--box] [--tee] [--skip_table] [--skip_pdf] [--skip_video] [--skip_trajopt]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --skip_table) DO_TABLE=false; shift ;;
        --skip_pdf) DO_PDF=false; shift ;;
        --skip_video) DO_VIDEO=false; shift ;;
        --skip_trajopt) SKIP_TRAJOPT=(--skip_trajopt); shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

status=()

run_step() {
    local label="$1"; shift
    echo ""
    echo "=============================================================="
    echo " $label"
    echo "   $*"
    echo "=============================================================="
    local t0 t1 rc
    t0=$(date +%s)
    set +e
    "$@"
    rc=$?
    set -e
    t1=$(date +%s)
    if (( rc == 0 )); then
        status+=("OK    $label ($((t1 - t0))s)")
    else
        status+=("FAIL  $label (exit $rc, $((t1 - t0))s)")
    fi
}

run_body() {
    local short="$1"
    local flag="--$short"
    if $DO_TABLE; then
        run_step "Benchmark table - $short" \
            bash planning_through_contact/scripts/ptc_benchmark.sh "$flag" --skip_short "${SKIP_TRAJOPT[@]}"
    fi
    if $DO_PDF; then
        run_step "PDF figures - $short" \
            bash planning_through_contact/scripts/ptc_render_figures.sh "$flag" --num 100 --max_paths 100
    fi
    if $DO_VIDEO; then
        run_step "Videos - $short" \
            bash planning_through_contact/scripts/ptc_visualize.sh "$flag" --num 100 --max_paths 100 --no_ground_truth
    fi
}

$RUN_BOX && run_body box
$RUN_TEE && run_body tee

echo ""
echo "Summary:"
for line in "${status[@]}"; do
    echo "  $line"
done
