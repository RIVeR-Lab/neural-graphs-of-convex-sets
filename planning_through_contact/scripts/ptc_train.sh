#!/usr/bin/env bash
# Train planar pushing flow GNN and RankNet checkpoints.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
PHASE="both"
CKPT_ROOT="planning_through_contact/checkpoints"
DATA_ROOT="$REPO_ROOT/planning_through_contact/dataset/data"
EXTRA_ARGS=()

usage() {
    echo "Usage: bash planning_through_contact/scripts/ptc_train.sh [--box] [--tee] [--phase flow|ranknet|both] [--data-root PATH] [--ckpt-root PATH] [extra train args...]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --phase) PHASE="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        flow|ranknet|both) PHASE="$1"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

if [[ "$PHASE" != "flow" && "$PHASE" != "ranknet" && "$PHASE" != "both" ]]; then
    echo "Unknown phase: $PHASE (use flow, ranknet, or both)" >&2
    exit 1
fi

train_body() {
    local body="$1"
    local data_dir="$DATA_ROOT/$body"
    local flow_ckpt="$REPO_ROOT/$CKPT_ROOT/${body}/${body}_flow.ckpt"

    echo ""
    echo "=== Planar Pushing Training ($body, phase=$PHASE) ==="
    echo "  Data dir    : $data_dir"
    echo "  Checkpoints : $CKPT_ROOT"
    echo ""

    if [[ "$PHASE" == "flow" || "$PHASE" == "both" ]]; then
        BODY="$body" \
        DATA_DIR="$data_dir" \
        bash planning_through_contact/scripts/planar_pushing/train_full_dataset.sh \
            --ckpt_dir "$CKPT_ROOT" \
            --experiment gcs_gnn \
            "${EXTRA_ARGS[@]}"
    fi

    if [[ "$PHASE" == "ranknet" || "$PHASE" == "both" ]]; then
        BODY="$body" \
        DATA_ROOT="$DATA_ROOT" \
        DATA_DIR="$data_dir" \
        FLOW_CKPT_PATH="$flow_ckpt" \
        bash planning_through_contact/scripts/planar_pushing/train_ranknet.sh \
            --ckpt_dir "$CKPT_ROOT" \
            --experiment ranknet \
            "${EXTRA_ARGS[@]}"
    fi
}

$RUN_BOX && train_body "sugar_box"
$RUN_TEE && train_body "tee"
