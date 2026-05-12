#!/usr/bin/env bash
# Generate dataset and train GNN + RankNet for one or both bodies.
#
# Usage:
#   bash scripts/generate_and_train.sh [--box] [--tee] [options]
#
# --box   Run for sugar_box (runs first if both specified)
# --tee   Run for tee
#
# Dataset size options (passed through to run_dataset_generation.sh):
#   NUM_TRAIN, NUM_VAL, NUM_TEST   (defaults: 500 / 100 / 100)
#   NUM_CANDIDATE_PATHS            (default: 100)
#
# Training options are passed through to the individual train scripts via extra args.
#
# Artifacts are written to the same paths used by download_data.sh:
#   planning_through_contact/dataset/data/<body>/   ← gcs_solutions.h5, global_features.csv, node_features.csv
#   checkpoints/gcs_gnn/<body>_flow.ckpt
#   checkpoints/ranknet/<body>_ranker.ckpt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        *) shift ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    echo "Usage: bash scripts/generate_and_train.sh [--box] [--tee]"
    echo "At least one of --box or --tee is required."
    exit 1
fi

NUM_TRAIN="${NUM_TRAIN:-500}"
NUM_VAL="${NUM_VAL:-100}"
NUM_TEST="${NUM_TEST:-100}"
NUM_CANDIDATE_PATHS="${NUM_CANDIDATE_PATHS:-100}"

run_pipeline() {
    local BODY="$1"

    echo ""
    echo "========================================"
    echo " Generate + Train — $BODY"
    echo "========================================"
    echo ""

    local DATA_DIR="$REPO_ROOT/planning_through_contact/dataset/data/$BODY"

    echo "--- Step 1: Dataset generation ---"
    BODY="$BODY" \
    DATA_DIR="$DATA_DIR" \
    NUM_TRAIN="$NUM_TRAIN" \
    NUM_VAL="$NUM_VAL" \
    NUM_TEST="$NUM_TEST" \
    NUM_CANDIDATE_PATHS="$NUM_CANDIDATE_PATHS" \
    bash planning_through_contact/dataset/run_dataset_generation.sh

    echo ""
    echo "--- Step 2: Train GNN flow predictor ---"
    BODY="$BODY" \
    DATA_DIR="$DATA_DIR" \
    bash planning_through_contact/scripts/planar_pushing/train_full_dataset.sh \
        --ckpt_dir checkpoints \
        --experiment gcs_gnn

    echo ""
    echo "--- Step 3: Train RankNet ranker ---"
    BODY="$BODY" \
    DATA_DIR="$DATA_DIR" \
    bash planning_through_contact/scripts/planar_pushing/train_ranknet.sh \
        --ckpt_dir checkpoints \
        --experiment ranknet

    echo ""
    echo "Artifacts for $BODY:"
    echo "  Data:       $DATA_DIR/"
    echo "  GNN ckpt:   checkpoints/gcs_gnn/${BODY}_flow.ckpt"
    echo "  Ranker ckpt: checkpoints/ranknet/${BODY}_ranker.ckpt"
}

$RUN_BOX && run_pipeline "sugar_box"
$RUN_TEE && run_pipeline "tee"
