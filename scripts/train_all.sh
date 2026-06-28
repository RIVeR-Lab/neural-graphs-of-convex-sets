#!/usr/bin/env bash
# Train all four GCS model families sequentially (Phase 1 flow GNN + Phase 2 RankNet each):
#   quadrotor/convex, quadrotor/nonlinear, manipulation/convex, manipulation/nonlinear
#
# Usage:
#   bash scripts/train_all.sh                 # all 4, defaults
#   bash scripts/train_all.sh --no_wandb      # extra flags forwarded to train_gcs_flow.py
#   COMBOS="manipulation:convex manipulation:nonlinear" bash scripts/train_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-python}
COMBOS=${COMBOS:-"quadrotor:convex quadrotor:nonlinear manipulation:convex manipulation:nonlinear"}
EXTRA_ARGS="$*"

for combo in $COMBOS; do
    problem="${combo%%:*}"
    planner="${combo##*:}"
    echo ""
    echo "=============================================================="
    echo "  Training ${problem} / ${planner}"
    echo "=============================================================="
    $PY scripts/train_gcs_flow.py \
        --problem "${problem}" \
        --planner "${planner}" \
        --phase both \
        ${EXTRA_ARGS}
done

echo ""
echo "All requested model families trained. Checkpoints under checkpoints/<problem>_<planner>/"
