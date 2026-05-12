#!/usr/bin/env bash
# Render paper-style trajectory PDF figures (ghosted slider + pusher trail, split by mode).
#
# Usage:
#   bash scripts/render_figures.sh [--box] [--tee] [--num N] [--max_paths N] [--without_ranknet]
#
# Defaults: 100 instances, 100 paths, 4 contact frames, 8 non-collision frames, with RankNet.
# Output: figures/<body>/figure_1/, figure_2/, etc.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
NUM=100
MAX_PATHS=100
WITHOUT_RANKNET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --num) NUM="$2"; shift 2 ;;
        --max_paths) MAX_PATHS="$2"; shift 2 ;;
        --without_ranknet) WITHOUT_RANKNET="--without_ranknet"; shift ;;
        *) shift ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

render_body() {
    local BODY="$1"
    echo ""
    echo "========================================"
    echo " Render figures — $BODY"
    echo "========================================"
    python3 planning_through_contact/scripts/planar_pushing/render_trajectory_figure.py \
        --body "$BODY" \
        --num "$NUM" \
        --max_paths "$MAX_PATHS" \
        $WITHOUT_RANKNET
}

$RUN_BOX && render_body "sugar_box"
$RUN_TEE && render_body "tee"
