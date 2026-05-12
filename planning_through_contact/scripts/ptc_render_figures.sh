#!/usr/bin/env bash
# Render planar pushing trajectory figures.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
NUM=100
MAX_PATHS=100
WITHOUT_RANKNET=()

usage() {
    echo "Usage: bash planning_through_contact/scripts/ptc_render_figures.sh [--box] [--tee] [--num N] [--max_paths N] [--without_ranknet]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --num) NUM="$2"; shift 2 ;;
        --max_paths) MAX_PATHS="$2"; shift 2 ;;
        --without_ranknet) WITHOUT_RANKNET=(--without_ranknet); shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

render_body() {
    local body="$1"
    echo ""
    echo "=== Planar Pushing Figures ($body) ==="
    python3 planning_through_contact/scripts/planar_pushing/render_trajectory_figure.py \
        --body "$body" \
        --output_dir planning_through_contact/results/figures \
        --num "$NUM" \
        --max_paths "$MAX_PATHS" \
        "${WITHOUT_RANKNET[@]}"
}

$RUN_BOX && render_body "sugar_box"
$RUN_TEE && render_body "tee"
