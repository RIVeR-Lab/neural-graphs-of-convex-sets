#!/usr/bin/env bash
# Benchmark Neural GCS vs Vanilla GCS on the quadrotor val (or test) set.
#
# Usage:
#   bash scripts/quadrotor_benchmark.sh [options]
#
# Options:
#   --h5_path PATH       HDF5 dataset path (default: quadrotor/dataset/quadrotor_gcs_dataset.h5)
#   --ckpt_dir DIR       Checkpoint root (default: quadrotor/checkpoints)
#   --output_dir DIR     Where to save benchmark_results.json (default: quadrotor/results)
#   --split SPLIT        Dataset split to evaluate: val or test (default: val)
#   --max_instances INT  Cap number of instances (default: all)
#   --device DEVICE      cuda or cpu (default: cuda)
#
# Example:
#   bash scripts/quadrotor_benchmark.sh
#   bash scripts/quadrotor_benchmark.sh --split test --device cpu

set -euo pipefail

H5_PATH="quadrotor/dataset/quadrotor_gcs_dataset.h5"
CKPT_DIR="quadrotor/checkpoints"
OUTPUT_DIR="quadrotor/results"
SPLIT="test"
MAX_INSTANCES=""
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --h5_path)       H5_PATH="$2";       shift 2 ;;
        --ckpt_dir)      CKPT_DIR="$2";      shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --split)         SPLIT="$2";         shift 2 ;;
        --max_instances) MAX_INSTANCES="$2"; shift 2 ;;
        --device)        DEVICE="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

FLOW_CKPT="$CKPT_DIR/quadrotor_gnn/quadrotor_flow_gnn.ckpt"
RANKNET_CKPT="$CKPT_DIR/quadrotor_ranknet/quadrotor_ranknet.ckpt"

EXTRA_ARGS=""
if [[ -n "$MAX_INSTANCES" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_instances $MAX_INSTANCES"
fi

# Include RankNet only if checkpoint exists
RANKNET_ARG=""
if [[ -f "$RANKNET_CKPT" ]]; then
    RANKNET_ARG="--ranknet_ckpt $RANKNET_CKPT"
fi

echo "=== Quadrotor Benchmark ==="
echo "  Dataset    : $H5_PATH"
echo "  Split      : $SPLIT"
echo "  Flow ckpt  : $FLOW_CKPT"
echo "  Output dir : $OUTPUT_DIR"
echo "  Device     : $DEVICE"
echo ""

python scripts/benchmark_quadrotor.py \
    --h5_path "$H5_PATH" \
    --flow_ckpt "$FLOW_CKPT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    $RANKNET_ARG \
    $EXTRA_ARGS