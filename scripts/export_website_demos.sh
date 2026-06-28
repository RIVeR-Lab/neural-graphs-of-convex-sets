#!/usr/bin/env bash
# Regenerate website demo MP4s with 2x overlay fonts and 16:9 output.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="/home/ananya/Code/website/Demo Videos"
EXPORT_DIR="$REPO_ROOT/website_exports"
PY="$REPO_ROOT/venv/bin/python"
DEVICE="${DEVICE:-cuda}"

cd "$REPO_ROOT"
mkdir -p "$EXPORT_DIR" "$WEB_DIR"

echo "==> Installing Playwright (if needed)..."
"$PY" -m pip install -q playwright
"$PY" -m playwright install chromium

record_html() {
  local html="$1"
  local mp4="$2"
  local duration="$3"
  local trim_start="${4:-0}"
  echo "==> Recording $mp4 (${duration}s, trim ${trim_start}s)..."
  "$PY" scripts/record_meshcat_html_to_mp4.py \
    "$html" "$mp4" \
    --width 3840 --height 2160 \
    --duration "$duration" \
    --trim-start "$trim_start" \
    --warmup 1.5
}

echo "==> Quadrotor: plan + render HTML..."
"$PY" scripts/demo_quadrotor_neural_motion.py --device "$DEVICE" --intro-hold-s 9
QUAD_HTML="$REPO_ROOT/quadrotor/results/motion_demo/motion_5x5_b42_q17.html"
record_html "$QUAD_HTML" "$EXPORT_DIR/Quadrotor.mp4" 20.6

echo "==> Pick and Place: physics demo + render HTML..."
"$PY" scripts/demonstrate_pick_and_place.py --device "$DEVICE"
PICK_HTML="$REPO_ROOT/manipulation/results/shelf_viz/blue_shelf_physics/pick_and_place.html"
record_html "$PICK_HTML" "$EXPORT_DIR/Pick and Place.mp4" 45.0

echo "==> Box Pushing: re-render HTML from saved trajectory..."
"$PY" scripts/demo_planar_pushing_neural_motion.py \
  --body sugar_box \
  --traj 94 \
  --render_only "$REPO_ROOT/planning_through_contact/results/motion_demo/motion_sugar_box_plan639_seed17_traj.pkl"
BOX_HTML="$REPO_ROOT/planning_through_contact/results/motion_demo/motion_sugar_box_plan639_seed17.html"
record_html "$BOX_HTML" "$EXPORT_DIR/Box Pushing.mp4" 37.0 45

echo "==> Tee Pushing: re-render HTML from saved trajectory..."
"$PY" scripts/demo_planar_pushing_neural_motion.py \
  --body tee \
  --traj 0 \
  --render_only "$REPO_ROOT/planning_through_contact/results/motion_demo/motion_tee_plan3_seed17_traj.pkl"
TEE_HTML="$REPO_ROOT/planning_through_contact/results/motion_demo/motion_tee_plan3_seed17.html"
record_html "$TEE_HTML" "$EXPORT_DIR/Tee Pushing.mp4" 31.867 45

echo "==> Copying to website..."
cp "$EXPORT_DIR/Quadrotor.mp4" "$WEB_DIR/Quadrotor.mp4"
cp "$EXPORT_DIR/Pick and Place.mp4" "$WEB_DIR/Pick and Place.mp4"
cp "$EXPORT_DIR/Box Pushing.mp4" "$WEB_DIR/Box Pushing.mp4"
cp "$EXPORT_DIR/Tee Pushing.mp4" "$WEB_DIR/Tee Pushing.mp4"

echo "Done. Website videos updated in: $WEB_DIR"
for f in "$WEB_DIR"/*.mp4; do
  echo "  $(basename "$f"): $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$f")"
done
