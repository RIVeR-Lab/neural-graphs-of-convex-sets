#!/usr/bin/env bash
# Visualize Neural GCS (GNN + RankNet) on test-set instances and save contact-annotated videos.
#
# Usage:
#   bash scripts/visualize.sh [--box] [--tee] [--num N] [--max_paths N] [--without_ranknet]
#
# --box              Run for sugar_box (default if neither specified)
# --tee              Run for tee
# --num N            Number of test instances to visualize (default: 4)
# --max_paths N      Rounding budget — 10 or 100 (default: 100)
# --without_ranknet  Use GNN only, no RankNet reranking (default: with RankNet)
# --no_ground_truth  Skip rendering ground_truth.mp4 (saves ~half the runtime)
#
# Outputs per instance are written to trajectories/gnn_run_<timestamp>_<body>/plan_<id>/trajectory/:
#   prediction.mp4     — Neural GCS rounded trajectory (green=contact, red=non-contact)
#   ground_truth.mp4   — Vanilla GCS trajectory for comparison (when H5 is present)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
NUM=1
MAX_PATHS=100
USE_RANKNET=true
RENDER_GT=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --num) NUM="$2"; shift 2 ;;
        --max_paths) MAX_PATHS="$2"; shift 2 ;;
        --without_ranknet) USE_RANKNET=false; shift ;;
        --no_ground_truth) RENDER_GT=false; shift ;;
        *) shift ;;
    esac
done

# Default to box if neither specified
if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

visualize_body() {
    local BODY="$1"
    local DATA_DIR="$REPO_ROOT/planning_through_contact/dataset/data/$BODY"
    local FLOW_CKPT="$REPO_ROOT/checkpoints/gcs_gnn/${BODY}_flow.ckpt"
    local RANKER_CKPT="$REPO_ROOT/checkpoints/ranknet/${BODY}_ranker.ckpt"

    echo ""
    echo "========================================"
    echo " Visualize — $BODY  (max_paths=$MAX_PATHS, ranknet=$USE_RANKNET)"
    echo "========================================"
    echo ""

    EXTRA_ARGS=()
    if $USE_RANKNET; then
        if [[ ! -f "$RANKER_CKPT" ]]; then
            echo "[warn] RankNet checkpoint not found: $RANKER_CKPT — falling back to GNN only."
        else
            EXTRA_ARGS+=(--ranker_ckpt_path "$RANKER_CKPT")
        fi
    fi
    if $RENDER_GT; then
        EXTRA_ARGS+=(--h5_path "$DATA_DIR/gcs_solutions.h5")
    fi

    python3 planning_through_contact/scripts/planar_pushing/plan_with_gnn.py \
        --body "$BODY" \
        --ckpt_path "$FLOW_CKPT" \
        --node_features_csv "$DATA_DIR/node_features.csv" \
        --plan_index_csv "$DATA_DIR/global_features.csv" \
        --use_test_plans \
        --max_test_plans "$NUM" \
        --max_paths "$MAX_PATHS" \
        --rounding_flow qp \
        --show_contact_legend \
        --video_width_px 1920 \
        --video_height_px 1080 \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

$RUN_BOX && visualize_body "sugar_box"
$RUN_TEE && visualize_body "tee"
