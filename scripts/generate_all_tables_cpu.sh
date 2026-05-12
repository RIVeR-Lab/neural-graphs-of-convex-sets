#!/usr/bin/env bash
# Generate metrics-table PDFs for all three experiments on CPU only.
#
# Three runs, each on the full 100-instance test split:
#   1) Quadrotor          (max_paths = 10, hardcoded in scripts/benchmark_quadrotor.py)
#   2) Planar pushing — sugar_box   (max_paths = 100, --skip_short, --skip_trajopt)
#   3) Planar pushing — tee         (max_paths = 100, --skip_short, --skip_trajopt)
#
# Outputs (auto-incremented per re-run so previous tables are preserved):
#   quadrotor/results/results_table.pdf                                 (overwrites in place)
#   benchmark_results/sugar_box_<N>/results_table.pdf
#   benchmark_results/tee_<N>/results_table.pdf
#
# Final-copy step (overwrite-each-time):
#   ~/quadrotor_results_table.pdf
#   ~/sugar_box_results_table.pdf
#   ~/tee_results_table.pdf
#
# Usage:
#   bash scripts/generate_all_tables_cpu.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

status=()

run_step() {
    local label="$1"; shift
    local cmd_str="$*"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo " ${label}"
    echo "   $ ${cmd_str}"
    echo "════════════════════════════════════════════════════════════════════════"
    local t0 t1 rc
    t0=$(date +%s)
    set +e
    bash -c "${cmd_str}"
    rc=$?
    set -e
    t1=$(date +%s)
    local elapsed=$((t1 - t0))
    if (( rc == 0 )); then
        echo "[ok] ${label} — ${elapsed}s"
        status+=("OK    ${label} (${elapsed}s)")
    else
        echo "[FAIL] ${label} — exit ${rc} after ${elapsed}s (continuing)"
        status+=("FAIL  ${label} (exit ${rc}, ${elapsed}s)")
    fi
}

OVERALL_T0=$(date +%s)

run_step "[1/3] Quadrotor — table (CPU, max_paths=10)" \
    "bash scripts/quadrotor_benchmark.sh --device cpu"

QUAD_SRC="$REPO_ROOT/quadrotor/results/results_table.pdf"
QUAD_DST="$HOME/quadrotor_results_table.pdf"
if [[ -f "$QUAD_SRC" ]]; then
    cp -v "$QUAD_SRC" "$QUAD_DST"
else
    echo "[warn] expected quadrotor table at $QUAD_SRC but it does not exist."
fi

run_step "[2/3] Planar pushing sugar_box — table (CPU, max_paths=100)" \
    "bash scripts/run_full_benchmark.sh --box --skip_short --skip_trajopt --device cpu"

# pick the most recently created sugar_box_<N> dir
BOX_DIR=$(ls -td "$REPO_ROOT"/benchmark_results/sugar_box_* 2>/dev/null | head -1 || true)
BOX_SRC="${BOX_DIR}/results_table.pdf"
BOX_DST="$HOME/sugar_box_results_table.pdf"
if [[ -n "$BOX_DIR" && -f "$BOX_SRC" ]]; then
    cp -v "$BOX_SRC" "$BOX_DST"
else
    echo "[warn] could not locate latest sugar_box results_table.pdf under benchmark_results/"
fi

run_step "[3/3] Planar pushing tee — table (CPU, max_paths=100)" \
    "bash scripts/run_full_benchmark.sh --tee --skip_short --skip_trajopt --device cpu"

TEE_DIR=$(ls -td "$REPO_ROOT"/benchmark_results/tee_* 2>/dev/null | head -1 || true)
TEE_SRC="${TEE_DIR}/results_table.pdf"
TEE_DST="$HOME/tee_results_table.pdf"
if [[ -n "$TEE_DIR" && -f "$TEE_SRC" ]]; then
    cp -v "$TEE_SRC" "$TEE_DST"
else
    echo "[warn] could not locate latest tee results_table.pdf under benchmark_results/"
fi

OVERALL_T1=$(date +%s)
TOTAL=$((OVERALL_T1 - OVERALL_T0))

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " Summary  (total: ${TOTAL}s)"
echo "════════════════════════════════════════════════════════════════════════"
for line in "${status[@]}"; do
    echo "  $line"
done
echo ""
echo " Tables copied to home folder:"
echo "   - $QUAD_DST"
echo "   - $BOX_DST"
echo "   - $TEE_DST"
echo ""
echo " Originals (with summary.json / per-instance data) still live at:"
echo "   - quadrotor/results/"
echo "   - benchmark_results/sugar_box_<N>/"
echo "   - benchmark_results/tee_<N>/"
