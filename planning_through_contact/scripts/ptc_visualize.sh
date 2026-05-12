#!/usr/bin/env bash
# Visualize Neural GCS planar pushing test-set instances.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
NUM=1
MAX_PATHS=100
USE_RANKNET=true
RENDER_GT=true

usage() {
    echo "Usage: bash planning_through_contact/scripts/ptc_visualize.sh [--box] [--tee] [--num N] [--max_paths N] [--without_ranknet] [--no_ground_truth]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --num) NUM="$2"; shift 2 ;;
        --max_paths) MAX_PATHS="$2"; shift 2 ;;
        --without_ranknet) USE_RANKNET=false; shift ;;
        --no_ground_truth) RENDER_GT=false; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

visualize_body() {
    local body="$1"
    local data_dir="$REPO_ROOT/planning_through_contact/dataset/data/$body"
    local flow_ckpt="$REPO_ROOT/planning_through_contact/checkpoints/${body}/${body}_flow.ckpt"
    local ranker_ckpt="$REPO_ROOT/planning_through_contact/checkpoints/${body}/${body}_ranker.ckpt"

    echo ""
    echo "=== Planar Pushing Visualization ($body) ==="
    echo "  Data dir  : $data_dir"
    echo "  Max paths : $MAX_PATHS"
    echo "  RankNet   : $USE_RANKNET"
    echo ""

    extra_args=()
    if $USE_RANKNET; then
        if [[ -f "$ranker_ckpt" ]]; then
            extra_args+=(--ranker_ckpt_path "$ranker_ckpt")
        else
            echo "[warn] RankNet checkpoint not found: $ranker_ckpt; using GNN only."
        fi
    fi
    if $RENDER_GT; then
        extra_args+=(--h5_path "$data_dir/gcs_solutions.h5")
    fi

    python3 planning_through_contact/scripts/planar_pushing/plan_with_gnn.py \
        --body "$body" \
        --output_dir planning_through_contact/results/trajectories \
        --ckpt_path "$flow_ckpt" \
        --node_features_csv "$data_dir/node_features.csv" \
        --plan_index_csv "$data_dir/global_features.csv" \
        --use_test_plans \
        --max_test_plans "$NUM" \
        --max_paths "$MAX_PATHS" \
        --rounding_flow qp \
        --show_contact_legend \
        --video_width_px 1920 \
        --video_height_px 1080 \
        "${extra_args[@]}"
}

$RUN_BOX && visualize_body "sugar_box"
$RUN_TEE && visualize_body "tee"
