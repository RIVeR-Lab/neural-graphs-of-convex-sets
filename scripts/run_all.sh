#!/usr/bin/env bash
# Full planar-pushing pipeline: benchmark table → trajectory PDFs → trajectory videos.
#
# Per body (--box and/or --tee), runs in sequence:
#   1) Table:  scripts/run_full_benchmark.sh --<body> --skip_short
#              100 test instances, all baselines (Vanilla GCS, Neural GCS ± RankNet,
#              contact-implicit). The "_short" 10-path variants are skipped.
#   2) PDFs:   scripts/render_figures.sh --<body> --num 100 --max_paths 100
#              GNN + RankNet only, paper-style trajectory figures (one PDF per instance).
#   3) Videos: scripts/visualize.sh --<body> --num 100 --max_paths 100 --no_ground_truth
#              GNN + RankNet only, no GCS ground-truth comparison video.
#
# Usage:
#   bash scripts/run_all.sh [--box] [--tee] [--skip_table] [--skip_pdf] [--skip_video] [--skip_trajopt]
#
# Defaults: --box only. Pass --tee alone to do tee only, or both flags for both.
# --skip_trajopt drops the SNOPT contact-implicit row from Step 1 (faster).
#
# Outputs:
#   benchmark_results/<body>_<N>/{summary.json, results_table.pdf}
#   figures/<body>/figure_<N>/plan_*/figure.pdf
#   trajectories/<body>/video_<N>/plan_*/trajectory/prediction.mp4

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_BOX=false
RUN_TEE=false
DO_TABLE=true
DO_PDF=true
DO_VIDEO=true
SKIP_TRAJOPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --box) RUN_BOX=true; shift ;;
        --tee) RUN_TEE=true; shift ;;
        --skip_table) DO_TABLE=false; shift ;;
        --skip_pdf)   DO_PDF=false;   shift ;;
        --skip_video) DO_VIDEO=false; shift ;;
        --skip_trajopt) SKIP_TRAJOPT="--skip_trajopt"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if ! $RUN_BOX && ! $RUN_TEE; then
    RUN_BOX=true
fi

step_status=()  # accumulator of "<body> step result"

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
        step_status+=("OK    ${label} (${elapsed}s)")
    else
        echo "[FAIL] ${label} — exit ${rc} after ${elapsed}s (continuing)"
        step_status+=("FAIL  ${label} (exit ${rc}, ${elapsed}s)")
    fi
}

run_body() {
    local SHORT="$1"          # "box" or "tee"
    local FLAG="--${SHORT}"

    if $DO_TABLE; then
        run_step "[1/3] Benchmark table — ${SHORT}" \
            "bash scripts/run_full_benchmark.sh ${FLAG} --skip_short ${SKIP_TRAJOPT}"
    fi
    if $DO_PDF; then
        run_step "[2/3] PDF figures — ${SHORT}" \
            "bash scripts/render_figures.sh ${FLAG} --num 100 --max_paths 100"
    fi
    if $DO_VIDEO; then
        run_step "[3/3] Videos — ${SHORT}" \
            "bash scripts/visualize.sh ${FLAG} --num 100 --max_paths 100 --no_ground_truth"
    fi
}

OVERALL_T0=$(date +%s)

$RUN_BOX && run_body box
$RUN_TEE && run_body tee

OVERALL_T1=$(date +%s)
TOTAL=$((OVERALL_T1 - OVERALL_T0))

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " Pipeline summary  (total: ${TOTAL}s)"
echo "════════════════════════════════════════════════════════════════════════"
for line in "${step_status[@]}"; do
    echo "  $line"
done
echo ""
echo " Outputs to look for:"
$RUN_BOX && echo "   - benchmark_results/sugar_box_<N>/{summary.json, results_table.pdf}"
$RUN_TEE && echo "   - benchmark_results/tee_<N>/{summary.json, results_table.pdf}"
$RUN_BOX && echo "   - figures/sugar_box/figure_<N>/plan_*/figure.pdf"
$RUN_TEE && echo "   - figures/tee/figure_<N>/plan_*/figure.pdf"
$RUN_BOX && echo "   - trajectories/sugar_box/video_<N>/plan_*/trajectory/prediction.mp4"
$RUN_TEE && echo "   - trajectories/tee/video_<N>/plan_*/trajectory/prediction.mp4"
