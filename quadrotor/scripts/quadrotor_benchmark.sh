#!/usr/bin/env bash
# Benchmark Neural GCS vs Vanilla GCS on the quadrotor val (or test) set.
#
# Usage:
#   bash quadrotor/scripts/quadrotor_benchmark.sh [options]
#
# Options:
#   --planner PLANNER    convex or nonconvex (default: convex)
#   --h5_path PATH       HDF5 dataset path (default: matches --planner)
#   --ckpt_dir DIR       Per-planner checkpoint directory
#   --output_dir DIR     Where to save results_table.pdf (default: quadrotor/results)
#   --split SPLIT        Dataset split to evaluate: val or test (default: test)
#   --max_instances INT  Cap number of instances (default: all)
#   --device DEVICE      cuda or cpu (default: cpu)
#
# Example:
#   bash quadrotor/scripts/quadrotor_benchmark.sh --planner convex
#   bash quadrotor/scripts/quadrotor_benchmark.sh --planner nonconvex --split test

set -euo pipefail

cd "$(dirname "$0")/../.."

PLANNER="convex"
CKPT_DIR=""
H5_PATH=""
OUTPUT_DIR="quadrotor/results"
SPLIT="test"
MAX_INSTANCES=""
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
    case $1 in
        --planner)       PLANNER="$2";       shift 2 ;;
        --h5_path)       H5_PATH="$2";       shift 2 ;;
        --ckpt_dir)      CKPT_DIR="$2";      shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --split)         SPLIT="$2";         shift 2 ;;
        --max_instances) MAX_INSTANCES="$2"; shift 2 ;;
        --device)        DEVICE="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$PLANNER" != "convex" && "$PLANNER" != "nonconvex" ]]; then
    echo "Unknown planner: $PLANNER (use convex or nonconvex)" >&2
    exit 1
fi

if [[ -z "$H5_PATH" ]]; then
    H5_PATH="quadrotor/dataset/quadrotor_gcs_${PLANNER}.h5"
fi

if [[ -z "$CKPT_DIR" ]]; then
    CKPT_DIR="quadrotor/checkpoints/quadrotor_${PLANNER}"
fi

TAG="quadrotor_${PLANNER}"
FLOW_CKPT="$CKPT_DIR/${TAG}_flow_gnn.ckpt"
RANKNET_CKPT="$CKPT_DIR/${TAG}_ranknet.ckpt"

EXTRA_ARGS=""
if [[ -n "$MAX_INSTANCES" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_instances $MAX_INSTANCES"
fi

RANKNET_ARG=""
if [[ -f "$RANKNET_CKPT" ]]; then
    RANKNET_ARG="--ranknet_ckpt $RANKNET_CKPT"
fi

echo "=== Quadrotor Benchmark ($PLANNER) ==="
echo "  Dataset    : $H5_PATH"
echo "  Split      : $SPLIT"
echo "  Flow ckpt  : $FLOW_CKPT"
echo "  RankNet    : ${RANKNET_CKPT} $([[ -f "$RANKNET_CKPT" ]] || echo '(missing)')"
echo "  Output dir : $OUTPUT_DIR"
echo "  Device     : $DEVICE"
echo ""

python3 quadrotor/scripts/benchmark_quadrotor.py \
    --planner "$PLANNER" \
    --h5_path "$H5_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --flow_ckpt "$FLOW_CKPT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    $RANKNET_ARG \
    $EXTRA_ARGS
