#!/usr/bin/env bash
# Generate planar pushing datasets for sugar_box and/or tee.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
NUM_TRAIN="${NUM_TRAIN:-500}"
NUM_VAL="${NUM_VAL:-100}"
NUM_TEST="${NUM_TEST:-100}"
NUM_CANDIDATE_PATHS="${NUM_CANDIDATE_PATHS:-100}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/planning_through_contact/dataset/data}"

usage() {
    echo "Usage: bash planning_through_contact/scripts/ptc_collect.sh [--box] [--tee] [--train-count N] [--val-count N] [--test-count N] [--candidate-paths N] [--data-root PATH]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --train-count) NUM_TRAIN="$2"; shift 2 ;;
        --val-count) NUM_VAL="$2"; shift 2 ;;
        --test-count) NUM_TEST="$2"; shift 2 ;;
        --candidate-paths) NUM_CANDIDATE_PATHS="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

collect_body() {
    local body="$1"
    local data_dir="$DATA_ROOT/$body"

    echo ""
    echo "=== Planar Pushing Data Collection ($body) ==="
    echo "  Data dir        : $data_dir"
    echo "  Counts          : train=$NUM_TRAIN val=$NUM_VAL test=$NUM_TEST"
    echo "  Candidate paths : $NUM_CANDIDATE_PATHS"
    echo ""

    BODY="$body" \
    DATA_DIR="$data_dir" \
    NUM_TRAIN="$NUM_TRAIN" \
    NUM_VAL="$NUM_VAL" \
    NUM_TEST="$NUM_TEST" \
    NUM_CANDIDATE_PATHS="$NUM_CANDIDATE_PATHS" \
    bash planning_through_contact/dataset/run_dataset_generation.sh
}

$RUN_BOX && collect_body "sugar_box"
$RUN_TEE && collect_body "tee"
