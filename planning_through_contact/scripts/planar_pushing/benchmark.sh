#!/usr/bin/env bash
set -euo pipefail

# Benchmark learned GCS planning against vanilla GCS on held-out test plans.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

BODY="${BODY:-sugar_box}"
MAX_TEST_PLANS="${MAX_TEST_PLANS:-10}"
FLOW_CKPT_PATH="${FLOW_CKPT_PATH:-$REPO_ROOT/checkpoints/gcs_gnn/${BODY}_flow.ckpt}"
RANKER_CKPT_PATH="${RANKER_CKPT_PATH:-$REPO_ROOT/checkpoints/ranknet/${BODY}_ranker.ckpt}"

ARGS=(
  --body "$BODY"
  --flow_ckpt_path "$FLOW_CKPT_PATH"
  --max_test_plans "$MAX_TEST_PLANS"
)

if [[ -f "$RANKER_CKPT_PATH" ]]; then
  ARGS+=(--ranker_ckpt_path "$RANKER_CKPT_PATH")
fi

exec python3 planning_through_contact/scripts/planar_pushing/benchmark_ranknet.py "${ARGS[@]}" "$@"
